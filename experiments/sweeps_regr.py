data = 'concrete'
data = 'boston'
data = 'diabetes'
coms = ''
for lr in [0.004, 0.04, 0.4]:
    for cycle_length in [10, 100, 1000]:
        for temperature in [50., 100., 200.]:
            coms += "python uci_csghmc.py --lr_0 %f --cycle_length %d  --offset %d  --epochs %d --temperature %f --data_name=%s\n" % (lr, cycle_length, cycle_length, cycle_length*10, temperature, data)
            
print('sweeps_%s.sh' % (data))
with open('sweeps_%s.sh' % (data), 'w') as f:
    f.write(coms)