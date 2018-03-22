awk '/impurity kin/ {e=$6} /helium ene/ {printf("%lf\t%lf\n",$2,$6+e)}' $1 
