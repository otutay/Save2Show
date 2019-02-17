# Save2Show
I'm using this class to save and show the data that is formed during training. 
First you need to create the hdf5 file with the given name
```
save = Save("fileName")
save.createFile()

```
Whenever you wanna save any data, you need to create it in the hdf5. Just ensure that first dimension of the data 1

```
save.createDataset(DataName1=Data1, DataName2=Data)
print("dataShape = ",Data1.shape)
```
```
dataShape = 1,XXX,XXX...
```

after that you can append new new results to that variable
```
save.variable2Write(DataName1=Data1, DataName2=Data)
```

Show.py reads the hdf5 file and plot the data with following given names. 
```
self.data2Plot2 = ['ValLoss',  'valTop1', 'valTop5']
self.data2Plot1 = ['loss', 'top1', 'top5']
```
each element in the data2Plot2 and data2Plot1 is plotted in the same graph.
You can use Show.py with the added hdf5 file.

just run
``` python Show.py```
