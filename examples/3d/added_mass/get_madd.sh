if [ $1 ]
then
	RANGE=$1
else
	RANGE=0.02
fi

gnuplot << EOF
data='<./get_data.sh'
E(x)=e0-n*x
set fit quiet
fit [0:$RANGE] E(x) data via e0,n
set xr [0:2.*$RANGE]
plot data, E(x)
pause 3
print "madd= ",n
EOF
