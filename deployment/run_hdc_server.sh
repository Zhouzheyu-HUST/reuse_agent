######### config #########
port=8710

######### run #########
hdc kill
hdc -s 0.0.0.0:"$port" -m
