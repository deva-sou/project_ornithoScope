# Get configs_list_path
configs_list_path=$1

# Check if the configs_list_path is a file
if [ ! -f "$configs_list_path" ]; then
    echo "'$configs_list_path' not found."
    exit 1
fi

it=0
while [ $(cat $configs_list_path | wc -l) -gt 0 ];
do
    it=$(($it + 1))

    # Extract first config_path from the list
    config_path=$(cat $configs_list_path | head -n 1)
    cat $configs_list_path | tail -n +2 > $configs_list_path'_tmp'
    mv $configs_list_path'_tmp' $configs_list_path

    # Check if the config_path is a file
    if [ ! -f "$config_path" ]; then
        echo "Config file '$config_path' does not exist."
        continue
    fi

    # Start training session with config_path
    waiting=true
    while $waiting
    do
        echo "Starting a new training session?"
        { # Try to start tmux 0
            tmux new-session -s 0 -d "source ../venv_ornithoscope/bin/activate ; export CUDA_VISIBLE_DEVICES=0 ; python3 train.py -c $config_path > logs/$it.log ; sleep 1000" &&
            waiting=false &&
            continue
        } || { # Try to start tmux 1
            tmux new-session -s 1 -d "source ../venv_ornithoscope/bin/activate ; export CUDA_VISIBLE_DEVICES=1 ; python3 train.py -c $config_path > logs/$it.log ; sleep 1000" &&
            waiting=false &&
            continue
        } || { # No one is free, wait
            echo "All sessions are busy. Retrying in 60s..."
            sleep 60
        }
    done
    echo "Training session started."
    echo
done