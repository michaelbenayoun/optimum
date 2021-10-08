#!/bin/bash
set -x

function main {
  init_params "$@"
  run_tuning
}

# init params
function init_params {
  tuned_checkpoint=saved_results
  topology=bert-base-uncased
  task_name=sst2
  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
           tuned_checkpoint=$(echo $var |cut -f2 -d=)
       ;;
      --task_name=*)
           task_name=$(echo $var |cut -f2 -d=)
       ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    extra_cmd='' 
    batch_size=64
    SCRIPTS=run_glue.py
    MAX_SEQ_LENGTH=128
    if [ ! $input_model ]; then
        model_name_or_path=$topology
    else
        model_name_or_path=$input_model
    fi

    python -u  run_glue.py \
        --model_name_or_path ${model_name_or_path} \
        --task_name ${task_name} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --output_dir ${tuned_checkpoint} \
        --per_device_eval_batch_size=${batch_size} \
        --per_device_train_batch_size=${batch_size} \
        --pad_to_max_length \
        --do_eval \
        --provider inc \
        --quantization_approach dynamic \
        --int8 \
        --quantize \

}

main "$@"
