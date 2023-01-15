
class KMeans():
    
    def __init__(self, read_addr, write_addr = None, sava_file_name = 'DEFAULT'):
        #try:
        self.read_addr = read_addr
        self.write_addr = write_addr
        if sava_file_name == 'DEFAULT':
            sfn = 'KMeans_Save_File'
        else:
            sfn = sava_file_name
        if self.write_addr is None:
            if path.exists('%s\\%s'%(self.read_addr, sfn)):
                rename('%s\\%s'%(self.read_addr, sfn), '%s\\%s%s'%(self.read_addr, sfn, time()))
            mkdir("%s\\%s"%(self.read_addr, sfn))
            self.write_addr = self.read_addr + '\\%s'%sfn
        else:
            if path.exists('%s\\%s'%(self.write_addr, sfn)):
                rename('%s\\%s'%(self.write_addr, sfn), '%s\\%s%s'%(self.write_addr, sfn, time()))
            mkdir("%s\\%s"%(self.write_addr, sfn))
            self.write_addr = self.write_addr + '\\%s'%sfn
        self.Vectors_Length = 'Length Of First Data'
        self.Number_Of_Centers = 100
        self.Number_Of_Levels  = 1000
        self.Show_Index_Center = False
        self.Dual_Process      = False
        self.F_Add_Centers     = False   #add centers in first level
        self.Add_Centers       = True
        self.__Auto_Run        = False
        self.Uniq_Mode         = True
        self.CFIDF             = False
        self.__vectors_name = []
        self.__concept = False
        self.__words_uniq = {}
        self.__Vectors = []
        self.__counter = 0
        self.__Words = []
        self.__ll = 0
        #except Exception as e:
         #   pass



    def __repr__(self):
        prnt = (self.Vectors_Length, self.Number_Of_Centers, str(self.Number_Of_Levels), self.Show_Index_Center, self.Dual_Process, self.Add_Centers, self.Uniq_Mode, self.CFIDF)
        return """\n
        Vectors_Length = %s\n
        Number_Of_Centers = %d\n
        Number_Of_Levels  = %s\n
        Show_Index_Center = %s\n
        Dual_Process      = %s\n
        Add_Centers       = %s\n
        Uniq_Mode         = %s\n
        CFIDF             = %s\n
        """%prnt



    def __check_data(self):
        try:
            if self.Dual_Process:
                self.number_of_levels = self.Number_Of_Levels
            if not self.Number_Of_Levels == 'ALL' and not self.Number_Of_Levels == 'FILE':
                self.__LL = self.Number_Of_Levels
            else:
                self.__LL = -1
            dir_name = listdir(self.read_addr)
            for i in dir_name :
                if i not in self.__vectors_name and i[-4:] == '.vec' :
                    file_object = open(self.read_addr + '\\' + i, 'r')
                    file_object.readline()
                    lines = file_object.readlines()
                    for line in lines:
                        temp = []
                        line = line.split()
                        for j in range(1,len(line)):
                            temp.append(float(line[j]))
                    if self.Vectors_Length == 'Length Of First Data':
                        self.__vectors_length = len(temp)
                        return True
                    else:
                        self.__vectors_length = self.Vectors_Length
                        return True
            if not self.F_Add_Centers:
                pass
            else:
                self.__f_vectors = []
                self.__f_words = []
                file_object = open(self.F_Add_Centers, 'r')
                file_object.readline()
                lines = file_object.readlines()
                for line in lines:
                    temp = []
                    line = line.split()
                    for j in range(1,len(line)):
                        temp.append(float(line[j]))
                    if len(temp) == self.__vectors_length:
                        self.__f_vectors.append(temp)
                        self.__f_words.append(line[0])
        except Exception as e:
            pass

    def __read_data(self, dirname = None):
        #try:
        a = 0
        if dirname is None:
            dir_name = listdir(self.read_addr)
            dir_name.sort(key=lambda x : len(x))
        else:
            dir_name = dirname
            file_object = open(dir_name, 'r')
            file_object.readline()
            lines = file_object.readlines()
            for line in lines:
                temp = []
                line = line.split()
                for j in range(1,len(line)):
                    temp.append(float(line[j]))
                if len(temp) == self.__vectors_length:
                    self.__Vectors.append(temp)
                    self.__Words.append(line[0])
                    i = ''
            for g in range(len(dir_name)-1, -1, -1):
                if dir_name[g : g+1] == '\\':
                    break
                i = dir_name[g] + i
            print('Vector [%s] Added To The Data .'%i)
            a += 1
            self.__vectors_name.append([i, len(lines)])
            return a
        for i in dir_name :
            if len(self.__vectors_name) == 0:
                if i[-4:] == '.vec' :
                    file_object = open(self.read_addr + '\\' + i, 'r')
                    file_object.readline()
                    lines = file_object.readlines()
                    for line in lines:
                        temp = []
                        line = line.split()
                        for j in range(1,len(line)):
                            temp.append(float(line[j]))
                        if len(temp) == self.__vectors_length:
                            self.__Vectors.append(temp)
                            self.__Words.append(line[0])
                    print('Vector [%s] Added To The Data .'%i)
                    a += 1
                    self.__vectors_name.append([i, len(lines)])
            else:
                if i not in [i[0] for i in self.__vectors_name] and i[-4:] == '.vec' :
                    file_object = open(self.read_addr + '\\' + i, 'r')
                    file_object.readline()
                    lines = file_object.readlines()
                    for line in lines:
                        temp = []
                        line = line.split()
                        for j in range(1,len(line)):
                            temp.append(float(line[j]))
                        if len(temp) == self.__vectors_length:
                            self.__Vectors.append(temp)
                            self.__Words.append(line[0])
                    print('Vector [%s] Added To The Data .'%i)
                    a += 1
                    self.__vectors_name.append([i, len(lines)])
        return a
        #except Exception as e:
         #   pass

    def run(self):
        #try:
        if not self.__concept:
            self.__check_data()
            self.__read_data()
        self.__dual = 0
        self.__concept_labels = []
        while True:
            if self.Dual_Process:
                if self.__dual == 0:
                    if self.__counter >= len(self.__vectors_name):
                        break
                    if self.__counter > 0:
                        self.__Centers = self.___Centers
                        self.__name_newcenter = self.___name_newcenter
                    self.Number_Of_Levels = 'FILE'
                    a = self.__levellize_data()
                    self.__dual = 1
                    print('\nlevellize_data (FILE) %s = %s'%(self.__counter+1, a))
                elif self.__dual == 1:
                    self.___Centers = self.__Centers
                    self.___name_newcenter = self.__name_newcenter
                    self.__counter -= 1
                    self.Number_Of_Levels  = 'ALL'
                    a = self.__levellize_data()
                    self.__dual = 0
                    print('\nlevellize_data (ALL) %s = %s'%(self.__counter+1, a))
            else:
                if self.Number_Of_Levels == 'ALL':
                    if self.__counter >= 1:
                        break
                elif self.Number_Of_Levels == 'FILE':
                    if self.__counter >= len(self.__vectors_name):
                        break
                elif self.__ll + self.__LL not in range(len(self.__Vectors)):
                    break
                a = self.__levellize_data()
                print('\nlevellize_data %s = %s'%(self.__counter+1, a))
            start = time()
            b = self.__km_processing()
            print('km_processing %s = %s'%(self.__counter+1, b))
            end = time()
            self.__time = end - start
            if self.__concept:
                self.__concept_labels.append(self.__Labels)
            else:
                c = self.__save_data()
                print('save_data %s = %s\n'%(self.__counter+1, c))
            self.__counter += 1
        if self.__Auto_Run:
            return
        print("k_means clustering done !")
       # except Exception as e:
       #     pass

    def auto_run(self, timee = None):
        if time == None:
            self.__Auto_Run = True
            while True:
                self.run()
        else:
            end = time() + float(timee)
            self.__Auto_Run = True
            while True:
                if time() >= end:
                    self.__Auto_Run = False
                    break
                self.run()

        print("k_means clustering done !")

    def __km_processing(self):
        #try:
        self.__vectors_uniq = {}
        for i in self.__words:
            if i in self.__vectors_uniq:
                self.__vectors_uniq[i] += 1
            else:
                self.__vectors_uniq[i] = 1 
            if i in self.__words_uniq:
                if self.__words_uniq[i][-1][0] == self.__counter:
                    self.__words_uniq[i][-1][1] += 1
                else:
                    self.__words_uniq[i].append([self.__counter, 1])
            else:
                self.__words_uniq[i] = [[self.__counter, 1]]

        if self.CFIDF:
            self.__CFIDF()

        DataSet = dict()
        for i in range(len(self.__words)):
            DataSet.update({self.__words[i]:self.__vectors[i]})
        self.__DataFrame = DF(DataSet, columns = self.__words)
        ###########
        import numpy as np
        import re
        from sklearn.preprocessing import normalize
        norm = np.linalg.norm(self.__DataFrame)
        normal_array = self.__DataFrame/norm
        ########
        self.__DataSet = self.__DataFrame.transpose()
        self.__DataSet.fillna(self.__DataSet.mean(), inplace=True)
        #K_M.euclidean_distances = self.__euc_dist
        kmeans = K_M.KMeans(algorithm='full', copy_x=True, init='k-means++', max_iter=300,n_clusters=self.Number_Of_Centers, n_init=10, n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001, verbose=0)
        kmeans.fit(self.__DataSet)
        centers = kmeans.cluster_centers_
        self.__Labels = list(kmeans.labels_)

        self.__Centers = []
        self.__name_newcenter = [None for i in range(self.Number_Of_Centers)]
        self.__centers_uniq = [{} for i in range(self.Number_Of_Centers)]
        for i in range(len(self.__Labels)):
            i_count = self.__Labels[i]
            if self.__words[i] in self.__centers_uniq[i_count]:
                self.__centers_uniq[i_count][self.__words[i]] += 1
            else:
                self.__centers_uniq[i_count][self.__words[i]] = 1
            if self.__name_newcenter[i_count] is None:
                self.__name_newcenter[i_count] = self.__words[i]

        for i in range(self.Number_Of_Centers):
            central = []
            for j in range(self.__vectors_length):
                central.append(centers[i][j])
            self.__Centers.append(central)

        if self.Uniq_Mode:
            self.__Centers = self.__Uniq_Mode(self.__Centers)

        self.__i_centers_name = []
        self.__i_centers_data = []
        for i in range(len(self.__Centers)) :
            self.__i_centers_name.append(['%s'%(self.__name_newcenter[i])])
            self.__i_centers_data.append(self.__Centers[i])
        return True
        #except Exception as e:
          #  pass

    def __levellize_data(self):
        #try:
        #DataSet = dict()
        if self.Number_Of_Levels == 'ALL':
            self.__LL = 0
            if self.Dual_Process:
                for i in range(self.__counter+1):
                    self.__LL += self.__vectors_name[i][1]
            else:
                for i in self.__vectors_name:
                    self.__LL += i[1]
            self.__words = self.__Words[:self.__LL]
            self.__vectors = self.__Vectors[:self.__LL]
            if not self.F_Add_Centers == False:
                self.__words = self.__f_words + self.__words
                self.__vectors = self.__f_vectors + self.__vectors
            #for i in range(len(self.__words)):
            #    DataSet.update({self.__words[i]:self.__vectors[i]})
            #self.__DataFrame = DF(DataSet, columns = self.__words)
            return True
        elif self.Number_Of_Levels == 'FILE':
            self.__LL = self.__vectors_name[self.__counter][1]
            self.__vec_name = self.__vectors_name[self.__counter][0]
        else:
            if self.Add_Centers:
                self.__LL = self.Number_Of_Levels - self.Number_Of_Centers
            else:
                self.__LL = self.Number_Of_Levels
            vec_size = 0
            for i in self.__vectors_name:
                vec_size += i[1]
                if self.__ll in range(vec_size):
                    self.__vec_name = i[0]
                    break
                
        if self.__counter == 0:
            if self.Number_Of_Levels == 'FILE':
                self.__vectors = self.__Vectors[self.__ll:self.__LL]
                self.__words  = self.__Words[self.__ll:self.__LL]
            elif not self.F_Add_Centers == False:
                self.__vectors = self.__f_vectors + self.__Vectors[self.__ll:self.__LL]
                self.__words = self.__f_words + self.__Words[self.__ll:self.__LL]
            else:
                self.__vectors = self.__Vectors[self.__ll:self.__LL + self.Number_Of_Centers]
                self.__words  = self.__Words[self.__ll:self.__LL + self.Number_Of_Centers]
            #for i in range(len(self.__words)):
            #    DataSet.update({self.__words[i]:self.__vectors[i]})
            #self.__DataFrame = DF(DataSet, columns = self.__words)
            if self.Number_Of_Levels == 'FILE':
                self.__ll += self.__LL
            else:
                self.__ll += self.__LL + self.Number_Of_Centers
            return True

        else :
            center_words = []
            if self.Show_Index_Center:
                for i in range(self.Number_Of_Centers):
                    center_words.append('%s(%s)'%(self.__name_newcenter[i], self.__counter))
            else:
                for i in range(self.Number_Of_Centers):
                    center_words.append('%s'%(self.__name_newcenter[i]))
            d_copy = self.__Vectors[self.__ll:self.__ll + self.__LL]
            w_copy = self.__Words[self.__ll:self.__ll + self.__LL]
            if self.Add_Centers:
                self.__vectors = self.__Centers + d_copy
                self.__words   = center_words + w_copy
            else:
                self.__vectors = d_copy
                self.__words   = w_copy
            #for i in range(len(self.__words)):
            #    DataSet.update({self.__words[i]:self.__vectors[i]})
            #self.__DataFrame = DF(DataSet, columns = self.__words)
            self.__ll += self.__LL
            return True
        #except Exception as e:
          #  return str(e) 
        

    def __save_data(self):
        #try:
        if self.Number_Of_Levels == 'ALL':
            mkdir('%s\\%s(ALL)'%(self.write_addr, self.__counter))
            s_adrr = '%s\\%s(ALL)'%(self.write_addr, self.__counter)
            if self.Dual_Process:
                for i in range(self.__counter+1):
                    copyfile('%s\\%s'%(self.read_addr, self.__vectors_name[i][0]), '%s\\%s'%(s_adrr, self.__vectors_name[i][0]))
            else:
                for i in self.__vectors_name:
                    copyfile('%s\\%s'%(self.read_addr, i[0]), '%s\\%s'%(s_adrr, i[0]))
        elif self.Number_Of_Levels == 'FILE':
            mkdir('%s\\%s(FILE)'%(self.write_addr, self.__counter))
            s_adrr = ('%s\\%s(FILE)'%(self.write_addr, self.__counter))
            if self.__concept and self.__counter == 0:
                pass
            else:
                copyfile('%s\\%s'%(self.read_addr, self.__vectors_name[self.__counter][0]), '%s\\%s'%(s_adrr, self.__vectors_name[self.__counter][0]))
        else:
            mkdir('%s\\%s'%(self.write_addr, self.__counter))
            s_adrr = ('%s\\%s'%(self.write_addr, self.__counter))
            vfile_out = open('%s\\%s-%s-EM.vec'%(s_adrr, self.__LL, self.__vectors_length), 'w')
            vfile_out.write("%s %s\n"%(self.__LL, self.__vectors_length))
            for i in range(self.__LL):
                vfile_out.write("%s %s\n"%(self.__words[i], self.__vectors[i]))
            vfile_out.close()
        if not (self.Number_Of_Levels == 'ALL' and self.Dual_Process):
            dflog = open('%s\\Centers.vec'%self.write_addr, 'w')
            dflog.write('%s %s\n'%(self.__LL, self.__vectors_length))
            for i in range(len(self.__i_centers_name)):
                dflog.write('%s'%self.__i_centers_name[i][0])
                for j in self.__i_centers_data[i]:
                    dflog.write(' %s'%j)
                dflog.write('\n')
            dflog.close()
        dflog = open('%s\\Data.txt'%s_adrr, 'w')
        dflog.write(str(self.__DataSet))
        dflog.close()
        
        b = 0
        texts_per_cluster=[0]*self.Number_Of_Centers
        for i in self.__Labels:
            dflog = open('%s\\cluster-%s.txt'%(s_adrr, i), 'a')
            dflog.write("\t" + self.__words[b] + "\n")
            dflog.close()
            b=b+1
            texts_per_cluster[i]+=1
                    
        dflog = open('%s\\info.txt'%s_adrr, 'w')
        if self.Number_Of_Levels == 'ALL':
            if self.Dual_Process:
                b = 0
                for i in self.__vectors_name:
                    b += i[1]
                    dflog.write('Vector %s Added To The Data .\n'%i[0])
                    if b <= self.__LL:
                        break
            else:
                for i in self.__vectors_name:
                    dflog.write('Vector %s Added To The Data .\n'%i[0])
        else:
            dflog.write('Vector %s Added To The Data .\n'%self.__vec_name)
        dflog.write("""
        Number_Of_Centers = %d\n
        Number_Of_Levels  = %s\n
        Show_Index_Center = %s\n
        Dual_Process      = %s\n
        Add_Centers       = %s\n
        Uniq_Mode         = %s\n
        CFIDF             = %s\n
                    """%(self.Number_Of_Centers, str(self.Number_Of_Levels), self.Show_Index_Center, self.Dual_Process, self.Add_Centers, self.Uniq_Mode, self.CFIDF))
        dflog.write('\n' + 'data_length : %s  \nDimention   : %s\n\ntime_left   : %s\n\n'%(self.__LL, self.__vectors_length, self.__time))

        b=0
        dflog.write("Clusters words count: "+"\n")
        for i_cluster in texts_per_cluster:
            dflog.write("\n"+"Cluster %s : %s " %(b,i_cluster))
            b=b+1
        dflog.close()
        
        file_object = open('%s\\Centers.csv'%s_adrr,'w')
        txt_object = open('%s\\Output.txt'%s_adrr, 'w')
        txt_object.write("Clustering Result %s : \n\n\n"%(self.__counter+1))
        for i in range(len(self.__i_centers_name)) :
            file_object.write(",%s"%(self.__i_centers_name[i][0]))
            txt_object.write("\tCenter[%s_%s][%s] = "%(self.__counter+1, i+1, self.__i_centers_name[i][0]) + "{} \n\n".format(self.__i_centers_data[i]))
            for j in range(len(self.__i_centers_data[i])) :
               file_object.write(",%s"%(self.__i_centers_data[i][j]))
            file_object.write("\n")
        file_object.close()
        txt_object.close()
        return True
            
        #except Exception as e:
            #pass

    def __euc_dist(X, Y=None, Y_norm=None, squared=False):
        #return pairwise_distances(X, Y, metric = 'cosine', n_jobs = 10)
        return 1 - cosine_similarity(X, Y)

    def __CFIDF(self):
        #try:
        for i in range(len(self.__words)):
            cf = self.__vectors_uniq[self.__words[i]] / self.__LL
            idf = log(self.__counter + 1 / len(self.__words_uniq[self.__words[i]]))
            if cf <= 0:
                cf = 0.00001
            elif idf <= 0:
                idf = 0.00001
            cfidf = cf * idf
            #print('cfidf = %s * %s = %s'%(cf, idf, cfidf))
            for j in range(self.__vectors_length):
                self.__vectors[i][j] *= cfidf
        return
       # except Exception as e:
        #    pass