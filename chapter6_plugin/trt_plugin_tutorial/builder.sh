export LD_LIBRARY_PATH=./LayerNormPlugin_base/:$LD_LIBRARY_PATH
python builder.py -x /home/br/program/bert_origin/bert_model.onnx -c /home/br/program/bert_origin/ -o ./model.plan -f | tee log.txt