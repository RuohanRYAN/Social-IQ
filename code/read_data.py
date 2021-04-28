import pickle
import os
loss_array = []
loss_array_dev = []
f_array = []
f_array_dev = []
acc_array = []
acc_array_dev = []

onlyfiles = [int(f.split("_")[1].split(".")[0]) for f in os.listdir("./model_data1") if os.path.isfile(os.path.join("./model_data1", f))]
onlyfiles.sort()
print(onlyfiles)
for i in range(0,len(onlyfiles)):
    with open("./model_data1/data_{}.pth".format(onlyfiles[i]), 'rb') as f:
        tmpdata = pickle.load(f)
        loss_array.append(tmpdata['tr_loss'])
        loss_array_dev.append(tmpdata['de_loss'])
        f_array.append(tmpdata['tr_f1'])
        f_array_dev.append(tmpdata['de_f1'])
        acc_array.append(tmpdata['tr_acc'])
        acc_array_dev.append(tmpdata['de_acc'])

print(max(acc_array_dev))
data = {'tr_loss':loss_array,'tr_f1':f_array,'tr_acc':acc_array,'de_loss':loss_array_dev,'de_f1':f_array_dev,'de_acc':acc_array_dev}
with open('model_data1.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
