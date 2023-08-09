import os
import ctypes
import numpy as np
from cuda import cudart  # cuda runtime API
import tensorrt as trt

soFilePath      = './LayerNorm.so'
nBS             = 4
nSL             = 128
nEmbedding      = 256
epsilon         = 6e-6

np.random.seed(97)

npToTRT = {np.int8:trt.int8,np.float16:trt.float16,np.int32:trt.int32,np.float32:trt.float32}
npToPFT = {np.int8:trt.PluginFieldType.INT8,np.float16:trt.PluginFieldType.FLOAT16,
            np.int32:trt.PluginFieldType.INT32,np.float32:trt.PluginFieldType.FLOAT32}



def check(a, b, weak = False):
    if weak:
        return np.all( np.abs(a - b) < epsilon)
    else:
        return np.all( a == b )

def layerNormCPU(bufferH):
    _x = bufferH[0]
    _0  = np.mean(_x,2)[:,:,np.newaxis]
    print(_0.shape)
    _1  = _x - _0
    _2  = _1 * _1
    _3  = np.mean(_2,2)[:,:,np.newaxis]       # standard deviation
    print(_3.shape)
    _4  = np.array(epsilon,dtype=np.float32)
    _5  = _4.reshape(1,1,1)
    print(_5.shape)
    _6  = _3 + _5
    _7  = np.sqrt(_6)
    _8  = 1 / _7                # 1/sqrt(...)
    _9  = _1 * _8
    return _9

def showPluginName():
    print("\n-----------------------current plugin begin---------------------\n\n")
    for c in trt.get_plugin_registry().plugin_creator_list:
        print(c.name)
    print("\n-----------------------current plugin end---------------------\n\n")

def getLayerNormPlugin():
    # plg_register = trt.get_plugin_registry()
    # plg_creator = plg_register.get_plugin_creator("LayerNorm", "1", "")
    # print(type(plg_creator))
    # if plg_creator is None:
    #     raise RuntimeError("could not find LayerNorm")
    # plugin = plg_creator.create_plugin("LayerNorm", trt.PluginFieldCollection([]))
    # print(type(plugin))
    # if plugin is None:
    #     raise RuntimeError("Could not create_plugin LayerNormPluginDynamic")

    for c in trt.get_plugin_registry().plugin_creator_list:
        # print(c.name)
        if c.name == 'LayerNorm_br':
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

def run():
    logger = trt.Logger(trt.Logger.VERBOSE)
    trt.init_libnvinfer_plugins(logger, '')
    handle = ctypes.cdll.LoadLibrary(soFilePath)
    if not handle:
        raise RuntimeError("Could not load plugin library. Is `LayerNorm.so` on your LD_LIBRARY_PATH?")

    builder         = trt.Builder(logger)
    network         = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config          = builder.create_builder_config()
    config.max_workspace_size = 6 << 30 #1024 * 1024 * 1024 # 6 << 30
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)
    # config.flags    = 0
    config.flags    = [0, 1 << int(trt.BuilderFlag.FP16)][0]

    inputTensorList = []
    inputTensorList.append( network.add_input('inputT', trt.float32, [-1,-1,256]) )
    print("inputTensorList: {}".format(inputTensorList))
    print(inputTensorList[0].name, inputTensorList[0].shape, inputTensorList[0].dtype)
    # inputTensorList.append( network.add_input('inputB', trt.float32, [256]) )
    # inputTensorList.append( network.add_input('inputA', trt.float32, [256]) )
    showPluginName()

    profile = builder.create_optimization_profile()
    profile.set_shape('inputT',[1,4,256],[1024,256,256],[1024,256,256])
    config.add_optimization_profile(profile)

    print("inputTensorList's length: {}".format(len(inputTensorList)))
    pluginLayer = network.add_plugin_v2(inputTensorList, getLayerNormPlugin())
    if pluginLayer is None:
        raise RuntimeError("add_plugin_v2() failed")
    
    print(pluginLayer.get_output(0).dtype)
    pluginLayer.get_output(0).dtype = [trt.float32, trt.float16][0]
    print(pluginLayer.get_output(0).dtype)

    network.mark_output(pluginLayer.get_output(0))

    print(type(network))
    print("network.num_layers: {}".format(network.num_layers))
    print(network.num_inputs)
    print(network.num_outputs)
    print(network.name)
    
    print("---------------build engine begin------------")
    engine = builder.build_engine(network, config)
    print("---------------build engine end------------")
    print(type(engine))

    print("---------------build serialized network begin------------")
    engineString = builder.build_serialized_network(network, config)
    print("---------------build serialized network end------------")
    print(type(engineString))
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0,[nBS,nSL,nEmbedding])
    print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    
    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))

    bufferH = []
    bufferH.append( np.random.rand(nBS,nSL,nEmbedding).astype(np.float32).reshape(nBS,nSL,nEmbedding) * 2 - 1)
    print(context.get_binding_shape(1))
    bufferH.append(np.empty(context.get_binding_shape(1),dtype=trt.nptype(engine.get_binding_dtype(1))))

    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    print("check result:")
    temp1 = bufferH[-1]
    temp2 = layerNormCPU(bufferH[:1])
    print(check(temp1,temp2,True), "max diff=%f"%(np.max(np.abs(temp1 - temp2))) )
    
    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == '__main__':
    os.system("rm -f ./*.trt")
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    run()