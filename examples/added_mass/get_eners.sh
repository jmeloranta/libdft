# v=0
for r in `ls res_* | sort -V`
do
	head -1 ${r} | awk '{printf $2"\t"}'
	tail -4 ${r} | awk '/ener/ {print $5}'
done
