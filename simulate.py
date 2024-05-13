
import numpy as np
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd



class simulate_prediction:
    def __init__(self, A=-6, B=0, C=1, a=1, b=1, c=1):
        self.A = A
        self.B = B
        self.C = C
        self.a = a
        self.b = b
        self.c = c
        return None

    def Data_generation(self,
                        r1=1,
                        r2=2,
                        r3=3,
                        source_num=500,
                        target_num=10,
                        test_num=200,
                        source_loc=5,
                        source_scale=3,
                        source_noise_scale=200,
                        target_uniform=5,
                        target_noise_scale=12):
        #=======Data generation===============
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        rng1 = np.random.RandomState(self.r1)
        X1 = rng1.normal(loc=source_loc,
                         scale=source_scale,
                         size=(source_num, 1))
        Y1 = self.A * X1 + self.B * np.power(X1, 2) + self.C * np.power(X1, 3)
        Y1 += rng1.normal(0, source_noise_scale, size=Y1.shape)
        rng2 = np.random.RandomState(self.r2)
        Xp = rng2.uniform(target_uniform,
                          -target_uniform,
                          size=(target_num, 1))
        Yp = self.a * Xp + self.b * np.power(Xp, 2) + self.c * np.power(Xp, 3)
        Yp += rng2.normal(0, target_noise_scale, size=Yp.shape)
        rng3 = np.random.RandomState(self.r3)
        Xtest = rng3.uniform(-target_uniform,
                             target_uniform,
                             size=(test_num, 1))
        Ytest = self.a * Xtest + self.b * np.power(
            Xtest, 2) + self.c * np.power(Xtest, 3)
        Ytest += rng3.normal(0, target_noise_scale, size=Ytest.shape)
        path_writer =  "baseline_simulate" + '.xlsx'
        sheet= 'plot'
        self.x_source = X1
        self.y_source = Y1
        self.x_target = pd.read_excel(path_writer, sheet_name=sheet).values[:, 5:6]
        self.y_target = pd.read_excel(path_writer, sheet_name=sheet).values[:, 6:7]
        self.x_test = Xtest
        self.y_test = Ytest

#=======Data normalization================

    def data_normalizaiton(self):
        self.std1 = StandardScaler()
        self.std2 = StandardScaler()
        self.std3 = StandardScaler()
        self.std4 = StandardScaler()
        data4 = np.hstack((self.x_source, self.y_source))
        data5 = np.hstack((self.x_target, self.y_target))
        self.x_source = self.std1.fit_transform(self.x_source)
        self.y_source = self.std2.fit_transform(self.y_source)
        self.x_target = self.std3.fit_transform(self.x_target)
        self.y_target = self.std4.fit_transform(self.y_target)
        self.x_sourcetarget = np.vstack((self.x_source, self.x_target))
        self.y_sourcetarget = np.vstack((self.y_source, self.y_target))
        x_test_real = self.x_test
        Y_test_real = self.a * x_test_real + self.b * np.power(
            x_test_real, 2) + self.c * np.power(x_test_real, 3)
        self.x_test = self.std3.transform(self.x_test)

        #======= save data========
        col_name_NOsource = ['x_source_normal', 'y_source_normal']
        col_name_NOtarget = ['x_target_normal', 'y_target_normal']
        col_name_source = ['x_source', 'y_source']
        col_name_target = ['x_target', 'y_target']
        col_name_REtest = ['x_test_real', 'y_test_true', 'y_test_real']
        data1 = np.hstack((self.x_source, self.y_source))
        data2 = np.hstack((self.x_target, self.y_target))
        data3 = np.hstack((
            x_test_real,
            self.y_test,
            Y_test_real,
        ))
        data1 = pd.DataFrame(data1, columns=col_name_NOsource)
        data2 = pd.DataFrame(data2, columns=col_name_NOtarget)
        data3 = pd.DataFrame(data3, columns=col_name_REtest)
        data4 = pd.DataFrame(data4, columns=col_name_source)
        data5 = pd.DataFrame(data5, columns=col_name_target)
        path_writer =  "original_datanew" + '.xlsx'
        writer = pd.ExcelWriter(path_writer)
        data1.to_excel(writer,
                       sheet_name='source_normal',
                       startcol=0,
                       index=False)
        data2.to_excel(writer,
                       sheet_name='target_normal',
                       startcol=0,
                       index=False)
        data3.to_excel(writer, sheet_name='test_real', startcol=0, index=False)
        data4.to_excel(writer,
                       sheet_name='source_real',
                       startcol=0,
                       index=False)
        data5.to_excel(writer,
                       sheet_name='target_real',
                       startcol=0,
                       index=False)
        writer.save()

if __name__ == '__main__':
    simulate_process = simulate_prediction(A=-5, C=2)

    simulate_process.Data_generation(r1=7,
                                        source_num=500,
                                        test_num=200,
                                        source_loc=7,
                                        source_scale=3,
                                        target_uniform=-4,
                                        source_noise_scale=200,
                                        target_noise_scale=6,
                                        target_num=10)
    simulate_process.data_normalizaiton()
