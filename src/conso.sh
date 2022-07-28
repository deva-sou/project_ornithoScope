while true
do
	# GPU consumptions
	power_1=$(nvidia-smi | grep -E -o '[0-9]+W' | awk 'NR==1' | grep -E -o '[0-9]+')
	power_2=$(nvidia-smi | grep -E -o '[0-9]+W' | awk 'NR==3' | grep -E -o '[0-9]+')

	# Time (ms)
	time_val=$(($(date +%s%N)/1000000))

	# Store in csv file
	echo $time_val,$power_1,$power_2 >> data/conso/conso.csv

	sleep 0.1
done
