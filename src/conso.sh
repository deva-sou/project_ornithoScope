while true
do
	power_1=$(nvidia-smi | grep -E -o '[0-9]+W' | awk 'NR==1' | grep -E -o '[0-9]+')
	power_2=$(nvidia-smi | grep -E -o '[0-9]+W' | awk 'NR==3' | grep -E -o '[0-9]+')
	time_val=$(($(date +%s%N)/1000000))

	echo $time_val,$power_1,$power_2 >> conso.log

	sleep 0.1
done
