import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import pdb

class Show():
    def __init__(self,fileName):
        self.fileName = fileName
        self.fid = 0
        self.dataLim = 64
        self.data2Plot2 = ['ValLoss',  'valTop1', 'valTop5']
        self.data2Plot1 = ['loss', 'top1', 'top5']
        self.axis = [(0,7),(0,100),(0,100)]

    def openFile(self):
        self.fid = h5py.File(self.fileName,'r')

    def closeFile(self):
        self.fid.close()

    def getHier(self):
        myHier = list(self.fid.keys())
        return myHier

    def extractData(self,param):
        data = np.array(self.fid[param])
        return data

    def plotNormal(self):
        self.openFile()
        # hier = self.getHier()
        # hier2Plot = self.dataNotContain(hier,'.')
        plt.figure()
        data = self.extractData('lr')
        # import pdb; pdb.set_trace()
        # data = data.view((-1,1))
        plt.plot(data)
        plt.savefig('lr.png',dpi=500)  
        
        for it in range(len(self.data2Plot1)):
            plt.figure()
            data1 = self.extractData(self.data2Plot1[it])
            data2 = self.extractData(self.data2Plot2[it])
            plt.plot(np.arange(data1.shape[0]),data1,label=self.data2Plot1[it])
            tempLoc = np.linspace(0,data1.shape[0],data2.shape[0])
            plt.plot(tempLoc,data2,label=self.data2Plot2[it])
            plt.ylim([self.axis[it][0],self.axis[it][1]])
            plt.legend(loc='best')
            # plt.subplot(2,1,2)
            plt.savefig(self.data2Plot1[it]+'.png',dpi=500)    
        self.closeFile()
        

    # def plotImageData(self):
    #     self.openFile()
    #     hier = self.getHier()
    #     hier2Plot = self.dataNotContain(hier,'.grad')
    #     hier2Plot = self.dataContain(hier2Plot,'.')

    #     for it,name in enumerate(hier2Plot):
    #         data = self.extractData(name)
    #         if self.dataContain(name,'b'):
    #             self.plotColorMapPlot(data,name,(data.shape[1],data.shape[0]))
    #         elif self.dataContain(name,'w'):
    #             tempData = data.reshape((data.shape[0],data.shape[1],data.shape[2]*data.shape[3]))
    #             tempData = tempData.mean(axis=2)
    #             self.plotColorMapPlot(tempData,name+'.mean',(tempData.shape[1],tempData.shape[0]))
    #             self.plotColorMapSubplots(data,name+'.all')

    # def plotColorMapSubplots(self,data,name2Save):
    #     tempShape = data.shape
    #     fig,ax = plt.subplots(tempShape[2],tempShape[3])
    #     for it in range(tempShape[2]):
    #         for ik in range(tempShape[3]):
    #             tempData = data[:,:,it,ik]
    #             # print(tempData.shape)
    #             im = ax[it,ik].imshow(tempData,aspect='auto',vmin=-0.2, vmax=0.2)
    #             # cbar = ax[it,ik].figure.colorbar(im, ax=ax[it,ik])
    #             plt.savefig(name2Save+'.png',dpi=500)





    # def plotColorMapPlot(self,data,name2Save,plotRange):
    #     fig,ax = plt.subplots()
    #     im = ax.imshow(data,aspect='auto',extent = [0, plotRange[0],0, plotRange[1]])
    #     ax.set_title(name2Save)
    #     cbar = ax.figure.colorbar(im, ax=ax)
    #     plt.savefig(name2Save+'.png',dpi=500)

    def divideData(self,data2Divide):
        tempSize = data2Divide.shape[1]
        chunkNum = np.ceil(tempSize/self.dataLim).astype(int)
        tempData = np.zeros((chunkNum,data2Divide.shape[0],self.dataLim))

        for it in range(chunkNum):
            if it != chunkNum-1:
                tempData[it,:,:] = data2Divide[:,it*64:(it+1)*64]
            else:
                sizeOfLast =  data2Divide[:,it*64:].shape[1]
                tempData[it,:,0:sizeOfLast] = data2Divide[:,it*64:]

        return tempData

    def dataContain(self,data,data2Find):
        tempOut = [ x  for x in data if x.find(data2Find)>-1]
        return tempOut

    def dataNotContain(self,data,data2Find):
        tempOut = [ x for x in data if x.find(data2Find)==-1]
        return tempOut


if __name__=="__main__":
    show = Show("yoloParameters.hdf5")
    show.plotNormal()
    # show.plotImageData()
