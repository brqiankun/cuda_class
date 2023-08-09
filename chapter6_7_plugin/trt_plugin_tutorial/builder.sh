export LD_LIBRARY_PATH=./LayerNormPlugin_base/:$LD_LIBRARY_PATH
python builder.py --fp16 -x /home/br/program/bert_origin/bert_model.onnx -c /home/br/program/bert_origin/ -o ./model_fp16.plan -f | tee log_fp16.txt

python builder.py -x /home/br/program/bert_origin/bert_model.onnx -c /home/br/program/bert_origin/ -o ./model_fp32.plan -f | tee log_fp32.txt