base=ebub_mass
vels=`seq 0.2 0.2 1.8`

for v in $vels 
do
	name=${base}_$v
	if [ ! -d $name ]
	then 
		mkdir $name
	fi
	sed s/VEL/$v/ base_added_mass.c > ${name}/added_mass.c
	cd $name
		make -f ../Makefile added_mass
		ln -s added_mass $name
		./$name > res_$v 2> errs_$v &
       cd ..	
done
echo 'Computing for v= '$vels
