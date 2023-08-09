import os
import ctypes
import numpy as np
from time import time_ns
import tensorrt as trt
from cuda import cudart

useFile         = False
soFilePath      = './LayerNorm.so'
nBS             = 4
nSL             = 256
nEmbedding      = 256
nTime           = 100
epsilon         = 1e-6

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
    # _x,b,a = bufferH
    # nEmbed = bufferH[0].shape[2]
    # _0  = np.mean(_x,2)[:,:,np.newaxis]
    # _1  = _x - _0
    # _2  = _1 * _1
    # _3  = np.mean(_2,2)[:,:,np.newaxis]
    # _4  = np.array(1e-12,dtype=np.float32)
    # _5  = _4.reshape(1,1,1)
    # _6  = _3 + _5
    # _7  = np.sqrt(_6)
    # _8  = 1 / _7                # 1/sqrt(...)
    # _9  = b
    # _10 = _9.reshape(1,1,nEmbed)
    # _11 = _8 * _10              # b/sqrt(...)
    # _12 = _0 * _11              # bμ/sqrt(...)
    # _13 = a
    # _14 = _13.reshape(1,1,nEmbed)
    # _15 = _14 - _12             # a-bμ/sqrt(...)
    # _16 = _x * _11              # bx/sqrt(...)
    # _17 = _15 + _16             # b(x-μ)/sqrt(...)+a
    # _18 = _17.reshape(bufferH[0].shape[0],bufferH[0].shape[1],bufferH[0].shape[2])
    # return _18
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


def getLayerNormPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'LayerNorm_br':
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

def run():
    testCase = "test<fp%s,bs=%d,sl=%d,nEmbed=%d>"%(['32','16'][0],nBS,nSL,nEmbedding)
    logger = trt.Logger(trt.Logger.VERBOSE)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    builder         = trt.Builder(logger)
    network         = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config          = builder.create_builder_config()
    config.max_workspace_size = 6 << 30
    config.flags    = [0,1<<int(trt.BuilderFlag.FP16)][0]

    inputTensorList = []
    inputTensorList.append( network.add_input('inputT', trt.float32, [-1,-1,256]) )
    inputTensorList.append( network.add_input('inputB', trt.float32, [256]) )
    inputTensorList.append( network.add_input('inputA', trt.float32, [256]) )

    profile = builder.create_optimization_profile()
    profile.set_shape('inputT',[1,4,256],[1024,256,256],[1024,256,256])
    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2(inputTensorList, getLayerNormPlugin())
    pluginLayer.get_output(0).dtype = [trt.float32,trt.float16][0]

    network.mark_output(pluginLayer.get_output(0))
    
    print(type(network))
    print("network.num_layers: {}".format(network.num_layers))
    print(network.num_inputs)
    print(network.num_outputs)
    print(network.name)

    print("---------------build engine begin------------")
    engine = builder.build_engine(network, config)
    print("---------------build engine end------------")
    engineString = builder.build_serialized_network(network, config)
    if engineString is not None:
        print("build engine done")
    else:
        raise RuntimeError("build engine failed")

    context = engine.create_execution_context()
    context.set_binding_shape(0,[nBS,nSL,nEmbedding])
    context.set_binding_shape(1,[nEmbedding])
    context.set_binding_shape(2,[nEmbedding])
    print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    # stream  = cuda.Stream()
    _, stream = cudart.cudaStreamCreate()
    assert _ == cudart.cudaError_t.cudaSuccess
    _, stream_flags = cudart.cudaStreamGetFlags(stream)
    print("stream_flags: {}".format(stream_flags))

    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    print("nInput: {}".format(nInput))
    nOutput = engine.num_bindings - nInput
    print("nOutput: {}".format(nOutput))
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))

    bufferH = []
    bufferH.append( np.random.rand(nBS,nSL,nEmbedding).astype(np.float32).reshape(nBS,nSL,nEmbedding) * 2 - 1)
    bufferH.append( np.ones(nEmbedding).astype(np.float32) )
    bufferH.append( np.zeros(nEmbedding).astype(np.float32) )
    bufferH.append(np.empty(context.get_binding_shape(3),dtype=trt.nptype(engine.get_binding_dtype(3))))

    bufferD = []
    for i in range(engine.num_bindings):
        # bufferD.append( cuda.mem_alloc(bufferH[i].nbytes) )
        print("bufferH[{}].nbytes: {}".format(i, bufferH[i].nbytes))
        _, dev_p = cudart.cudaMalloc(bufferH[i].nbytes)
        assert _ == cudart.cudaError_t.cudaSuccess
        print(dev_p)
        bufferD.append(dev_p)


    for i in range(nInput):
        # cuda.memcpy_htod_async(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)), stream)
        print("H2D : cudaMemcpyAsync(bufferH[{}])------".format(i))
        _ = cudart.cudaMemcpyAsync(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        assert _[0] == cudart.cudaError_t.cudaSuccess
    # context.execute_async_v2(bufferD, stream.handle)
    context.execute_async_v2(bufferD, stream_flags)
    # stream.synchronize()
    cudart.cudaStreamSynchronize(stream)

    for i in range(nOutput):
        _ = cudart.cudaMemcpyAsync(bufferH[nInput+i].data, bufferD[nInput+i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        assert _[0] == cudart.cudaError_t.cudaSuccess
        # cuda.memcpy_dtoh_async(bufferH[nInput+i], bufferD[nInput+i], stream)
    cudart.cudaStreamSynchronize(stream)
    
    for i in range(nInput):
        temp = bufferH[i]
        print("inputH%d"%i, temp.shape,np.sum(abs(temp)),np.var(temp),np.max(temp),np.min(temp),np.sum(np.abs(np.diff(temp.reshape(-1)))))
        print(temp.reshape(-1)[:10])
        #print(temp)
    
    for i in range(nOutput):
        temp = bufferH[nInput+i]
        print("outputH%d"%i, temp.shape,np.sum(abs(temp)),np.var(temp),np.max(temp),np.min(temp),np.sum(np.abs(np.diff(temp.reshape(-1)))))
        #print(temp)
    

    for i in range(10):
        # context.execute_async_v2(bufferD, stream.handle)
        context.execute_async_v2(bufferD, stream_flags)
    # stream.synchronize()
    cudart.cudaStreamSynchronize(stream)
            
    time0 = time_ns()
    for i in range(nTime):
        # context.execute_async_v2(bufferD, stream.handle)
        context.execute_async_v2(bufferD, stream_flags)
    # stream.synchronize()
    cudart.cudaStreamSynchronize(stream)
    time1 = time_ns()
    print(testCase+"average %fms per inference\n"%((time1-time0)/nTime/1000000))


    print("check result:")
    temp1 = bufferH[-1]
    temp2 = layerNormCPU(bufferH[:3])

    print(check(temp1,temp2,True), "max diff=%f"%(np.max(np.abs(temp1 - temp2))) )

if __name__ == '__main__':
    os.system("rm -f ./*.trt")
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)

    run()

    #print("test all finish!")
