IMAGES="/media/florian/Penelope/Kaggle/Fishy/data/test_stg1_resize/*.jpg"
for file in $IMAGES
do
	echo "$file"
	convert $file -resize "448x448^" -gravity center -crop 448X448+0+0 +repage $file

done
