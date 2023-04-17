data = 'mnist'
data = 'ionosphere'
# data = 'parkinsons'

coms = ''
for lr in [0.1, 0.4, 0.8]:
    for cycle_length, offset in [(10, 3)]:
        for temperature in [1.]:
            for num_hidden in [32]:
                for weight_decay in [1e-4]:
                    coms += "python uci_csghmc.py --regress 0 --lr_0 %f --cycle_length %d  --offset %d  --epochs %d --temperature %f --weight_decay %f --data_name %s --num_hidden %i\n" % (lr, cycle_length, offset, cycle_length*10, temperature, weight_decay, data, num_hidden)
            
with open('sweeps_%s.sh' % (data), 'w') as f:
    f.write(coms)