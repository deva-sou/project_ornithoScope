path_to_images='/home/acarlier/code/data_ornithoscope/p0133_bird_data/raw_data'
path_to_all_images='/home/acarlier/code/project_ornithoScope/src/data/inputs/input_true_all.csv'
# Generate new data

for file in $(find $path_to_images | grep -v 'detected\|iNat')
do 
if [ -d $file ]
then
    continue
else
#echo $file
a=${file##*raw_data/}
echo $a>>$path_to_all_images
fi
done