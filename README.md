# 2021 Machine Learning Final Project - Forest fire detection
0816064 吳中赫
##### p.s. 此為test data 請下載後再運行程式：
https://drive.google.com/drive/folders/1tlOIF-eUURTEMBceirZ5pUg8QxYRVPgg?usp=sharing
## I. Introduction
有鑑於近幾年，氣候變化越發劇烈，包括澳洲、阿爾及利亞、葡萄牙、加州等地點，發生森林火災的頻率也增加不少，所以我想利用現有的森林火災照片資料，實做一個realtime的森林火災偵測模型，只要將監控設備架在森林裡，這個模型會根據回傳影片的每個frame去預測目前可能發森森林火災的機率，以達到即時發現森林火災發生，並趕快派消防隊前往。
會想要做這個的原因，其實是因為比起可能用什麼乾燥度、濕度、風向等等，火真的是用視覺來觀察會快也最直覺，但也不可能都派一個人在森林裡顧有沒有失火，所以用機器學習的方法解決感覺會很實用，而我也相信我這次實做出來的東西可以在現實中派上用場，而且其實也可以用再家用等其他地方(加多一點training data)，所以是個有趣又實用的題目。
![](https://i.imgur.com/y5q75w7.gif)


## II. Data Collection
* 因為主要都是image data，所以除了有找到一個比較大的forest fire image dataset以外，我還有自己在其他網站挑一些我覺得適合的照片放進我的dataset，照片主要分為以下三類：
    * 包括正常森林沒有失火的照片。
    * 沒有拍到火源但有明顯煙霧的照片。
    * 森林失火的照片
* application需要用到的video data，就是再以下網站找
* 以下為照片來源來源網站。
https://data.mendeley.com/datasets/gjmr63rz2r/1
https://www.pexels.com/zh-tw/search/forest%20fire/
## III. Preprocessing
1. 首先，先將data的資料夾分成以下結構，以方便我做後續處理。
    ```
    Dataset
    └───Testing
        └───fire
        └───nofire
    └───Training and Validation
        └───fire
        └───nofire
    ```
2. fire, nofire個別先挑一張照片，並用cv的技巧把他show出來確認一下，值得一提的是，再show出來前要先將照片顏色從BGR換成RGB不然出來顏色會怪怪的，而且在之後做predict時，照片也要記得做這件事，不然會不準確。
![](https://i.imgur.com/51ViC0T.png)
![](https://i.imgur.com/xG5JHC8.png)
3. 正式進入preprocessing，因為我總共用了SVM, Random Forest, CNN，三種model，但前兩種的用sklearn實作，CNN則是用pytorch實作，所以需要的前處理不太一樣，以下分為兩部分來講

    ### SVM, Random Forest Preprocessing
    1. 因為sklearn的模型，可以直接丟pd.DataFrame()進去，所以我將照片統一轉為DataFrame的格式。
    2. 所以首先，先將照片用 **cv2.imread()** 讀成array，並將其resize成 **(224, 224, 3)** ，最後將shape是 **(224, 224, 3)** 大小的，flat成一個一維的array。
    3. 接著，因為剛剛已經分好fire, nofire的資料夾了，所以就根據這張image的來源，紀錄image label。
    4. 所以最後就得到每張照片preprocessing後的array，將這些全部array記錄起來，轉成DataFrame的格式。
    5. 用以上步驟對train image, test image各做一遍，最後得到df_train, df_test，以下為 **df_train** ， **shape = (1527, 150529)** 。 
    ```
	0	1	2	3	4	5	6	7	8	9	...	150519	150520	150521	150522	150523	150524	150525	150526	150527	label
    0	0.037742	0.038652	0.040433	0.049678	0.059899	0.078702	0.005660	0.007745	0.044874	0.006422	...	0.075018	0.067174	0.145606	0.078431	0.070588	0.149020	0.078431	0.070588	0.149020	fire
    1	0.012290	0.016387	0.100560	0.012369	0.016466	0.102648	0.013751	0.017980	0.113690	0.014803	...	0.092980	0.038078	0.085137	0.090879	0.035977	0.083036	0.090196	0.035294	0.082353	fire
    2	0.109655	0.184165	0.333185	0.116948	0.192744	0.341763	0.115467	0.197496	0.346516	0.124300	...	0.098613	0.137829	0.177044	0.099685	0.142139	0.174877	0.099685	0.142822	0.174195	fire
    3	0.002276	0.006893	0.058972	0.002197	0.007108	0.058623	0.001821	0.007108	0.058867	0.001821	...	0.023529	0.015686	0.062745	0.023529	0.015686	0.062745	0.023529	0.015686	0.062745	fire
    4	0.763796	0.701050	0.650070	0.764439	0.701694	0.650713	0.767556	0.704810	0.653830	0.769218	...	0.099150	0.224156	0.275298	0.152514	0.268169	0.325667	0.165297	0.279022	0.337846	fire
    ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
    1522	0.191661	0.179897	0.160342	0.120200	0.108436	0.089580	0.149911	0.138147	0.118539	0.168118	...	0.135388	0.194344	0.162905	0.206239	0.258585	0.227212	0.166916	0.217896	0.186524	nofire
    1523	0.917901	0.925744	0.925744	0.907607	0.915450	0.915450	0.882884	0.890727	0.890727	0.854819	...	0.141356	0.239395	0.223708	0.054237	0.152276	0.136590	0.057686	0.155725	0.140039	nofire
    1524	0.319438	0.678174	0.580817	0.263081	0.578086	0.487115	0.284530	0.532206	0.453870	0.325625	...	0.198245	0.188126	0.144989	0.194366	0.193001	0.146625	0.241824	0.249212	0.202153	nofire
    1525	0.980392	0.984314	0.976471	0.980392	0.984314	0.976471	0.980392	0.984314	0.976471	0.980392	...	0.068540	0.060697	0.056775	0.073880	0.066036	0.062115	0.067857	0.060014	0.056092	nofire
    1526	0.991260	0.928515	0.904985	0.982038	0.919293	0.895763	0.968974	0.906229	0.882699	0.959191	...	0.124227	0.195769	0.184004	0.085295	0.142325	0.133799	0.156968	0.208176	0.200333	nofire
    1527 rows × 150529 columns
    ```

    6. 接著用train_test_split()，將剛剛得到的df_train,分成train和validation，再個別分成X, y，這樣preprocessing就完成了。
    ### CNN Preprocessing
    1. Pytorch的模型Input需要dtype = tensor，所以我們要先將image轉成tensor。
    2. 首先用 **torchvision.transforms.Compose** 將Train image, Test image轉換成shape = (224, 224, 3),再用 **torchvision.datasets.ImageFolder** 分好fire, nofire，得到train_val_data, test_data。
    3. 再用 **torch.utils.data.random_split()** 把剛剛的train_val_data分成train_data, val_data，(其實就是train_test_split)。
    4. 最後用 **torch.utils.data.dataloader()** 將剛剛的train_data, val_data, test_data，轉為一個一個batch的dataloader形式，並shuffle。這樣CNN Preprocessing就完成了。
    
## IV. Models
* ### Random Forest
    用sklearm的 RandomForestClassifier建立n_estimator=30(也就是5棵樹)，max_depth=30的Random Forest。
* ### SVM
    用sklearn的SVC，kernel=linear建立，SVM Classifier模型。
* ### CNN
    #### Model
    1. 因為Foreset Fire Detection，其實就是將影片裡的每個frame做fire, nofire的分類，所以比起一般的ANN我採用適合圖片的CNN，而我使用的框架是pytorch。
    2. 首先建立一個Class CNNModel，並繼承pytorch nn.Module可以用的function，接著以下分為兩部分：
        * Init：建立之後forward會用到那些層和激發函數。
            * 因為原本照片的input shape = (224, 224, 3)，所以一開始Convolutional layer的input channel = 3
            * 接著做激發函數，再來做maxpooling，用意是將進來的input縮小，有抗雜訊的功用。
            * 之後再做第二層CNN，最後用nn.Linear輸出平滑的結果，這個結果會是ex.[4, -3]，意思就是取大的就是那一個index當作預測結果，而nn.Linear裡的參數(8*50*50)，也就是經過上面所有計算得出的shape。
            ```
            def __init__(self):
                    super(CNN_Model, self).__init__()

                    self.cnn1 = nn.Conv2d(3, 16, kernel_size=5, stride = 1)
                    self.relu1 = nn.ReLU(inplace=True)
                    self.maxpool1 = nn.MaxPool2d(kernel_size=2)

                    self.cnn2 = nn.Conv2d(16, 8, kernel_size=11, stride = 1)
                    self.relu2 = nn.ReLU(inplace=True)
                    self.maxpool2 = nn.MaxPool2d(kernel_size=2)

                    self.fc = nn.Linear(8*50*50, 2)
            ```
            
        * forward：進來的Input要怎麼走。
            ```
            def forward(self, x):
                x = self.cnn1(x)
                x = self.relu1(x)
                x = self.maxpool1(x)
                x = self.cnn2(x)
                x = self.relu2(x)
                x = self.maxpool2(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
            return x
            ```
    3. 接著因為Pytorch有提供可以用GPU做運算，所以使用**model.to(device)**，device則是當可以使用GPU時就是GPU，不能使用時就用CPU。
    #### Train, Validation
    1. 首先，因為是做classification，所以採用先設定**loss_fn = nn.CrossEntropy**，**optimizer**則採用Adam。
    2. 接著用**def train**，把剛剛preprocessed data，以batch為單位進行train，再用**def val**進行驗證。
    3. 以epoch = 5，訓練剛剛的資料集。
## V. Application
application主要是把我的Web Cam真的當作放在森林裡的相機，但因為我身邊沒有真的森林所以除了真的打開電腦的相機以外，還可以用將影片輸入的方式，去判斷這個影片是否有forest fire的情況。
1. 用cv2.VideoCapture將影片讀入，然後再以read()的方式將影片變成一張一張的frame，而這時的frame其實就是一張照片，那我們將照片再用上述CNN Preprocessing的方式，轉成pytorch要求的格式，這時候就可以丟進model predict了。
2. predict完後，將結果丟入softmax()轉成相加機率為一的結果，並取機率較大的label就會是我們預測的結果。
3. 將最終得到的機率、label，用cv2.putText()放在每張frame上，接著用cv2.imshow()，就可以印出一張張frame，其實就是撥放影片了。
4. 最後將這個影片存成.mp4檔就會得到結果。
## VI. Results
Result的部分，分別以validation data, test data, forest fire video, web cam, 四個部分做比較。而模型如果採用Random Forest和SVM，因為在input傳進傳前，要先做較大量的preprocessing，才能正常運行，而這會耗費大量時間導致每張frame會顯示得特別慢，也就是影片會lag，所以就不做forest fire video, web cam的result。
* ### SVM
```
##### SVM Validation #####
+--------+--------------------+--------------------+--------------------+
| label  |      Accuracy      |     Precision      |       Recall       |
+--------+--------------------+--------------------+--------------------+
|  fire  | 0.9385964912280702 | 0.9344262295081968 |        0.95        |
| nofire | 0.9385964912280702 | 0.9433962264150944 | 0.9259259259259259 |
+--------+--------------------+--------------------+--------------------+
     0    1
0  228   12
1   16  200
##### SVM Test #####
+--------+-------------------+--------------------+--------------------+
| label  |      Accuracy     |     Precision      |       Recall       |
+--------+-------------------+--------------------+--------------------+
|  fire  | 0.868421052631579 | 0.8535353535353535 | 0.8894736842105263 |
| nofire | 0.868421052631579 | 0.8846153846153846 | 0.8473684210526315 |
+--------+-------------------+--------------------+--------------------+
     0    1
0  169   21
1   29  161
``` 
* ### Random Forest
```
##### Random Forest Validation #####
+--------+--------------------+--------------------+--------------------+
| label  |      Accuracy      |     Precision      |       Recall       |
+--------+--------------------+--------------------+--------------------+
|  fire  | 0.9013157894736842 | 0.925764192139738  | 0.8833333333333333 |
| nofire | 0.9013157894736842 | 0.8766519823788547 | 0.9212962962962963 |
+--------+--------------------+--------------------+--------------------+
     0    1
0  212   28
1   17  199
##### Random Forest Test #####
+--------+--------------------+--------------------+--------------------+
| label  |      Accuracy      |     Precision      |       Recall       |
+--------+--------------------+--------------------+--------------------+
|  fire  | 0.8236842105263158 | 0.8435754189944135 | 0.7947368421052632 |
| nofire | 0.8236842105263158 | 0.8059701492537313 | 0.8526315789473684 |
+--------+--------------------+--------------------+--------------------+
     0    1
0  151   39
1   28  162
```
* ### CNN
```
##### CNN Validation #####
+--------+--------------------+--------------------+--------------------+
| label  |      Accuracy      |     Precision      |       Recall       |
+--------+--------------------+--------------------+--------------------+
|  fire  | 0.9320175438596491 | 0.8863636363636364 | 0.9957446808510638 |
| nofire | 0.9320175438596491 | 0.9947916666666666 | 0.8642533936651584 |
+--------+--------------------+--------------------+--------------------+
     0    1
0  234    1
1   30  191
##### CNN Test #####
+--------+--------------------+--------------------+--------------------+
| label  |      Accuracy      |     Precision      |       Recall       |
+--------+--------------------+--------------------+--------------------+
|  fire  | 0.8868421052631579 | 0.8237885462555066 | 0.9842105263157894 |
| nofire | 0.8868421052631579 | 0.9803921568627451 | 0.7894736842105263 |
+--------+--------------------+--------------------+--------------------+
     0    1
0  187    3
1   40  150
```
* ##### 那我們隨機挑兩張照片來看看預測結果如何!
![](https://i.imgur.com/51ViC0T.png)
`The image is predicted to be 99.48% fire`
![](https://i.imgur.com/xG5JHC8.png)
`The image is predicted to be 99.94% nofire`
以上的x% fire/ no fire，則是在把predict後的輸出丟進softmax()裡，他會轉換成相加=1的機率，再取大的就行了。
* ##### 接著就進入我們的重頭戲，video detection 和webcam detection 的結果! 
    https://youtu.be/cfWlatcAi2I (這個因為用了Discovery的素材所以可能開不了)
    https://youtu.be/riqoDSZ5Hs8 (可以幫我留言按讚)
## VII. Conclusion
* 三個model的比較：
    * CNN在三個model中無庸置疑是最準的，而SVM又比Random Forest好一些，我這次採用的SVM kernel = linear，所以如果是用kernel改用poly搭配grid search，我猜效果一定會更好，但礙於時間一定會拉得很長所以就沒有採用了。
    * 除此之外，如果選用SVM或Random Forest，前處理需要將三維數值先resize，然後在flat後，轉成df，preprocessing的時間高得嚇人，沒辦法即時的predict每個frame，但這就違背本次project想要做到的即時detection。再加上sklearn好像又不支援用GPU跑，可見Pytorch CNN的好啊!
* How robust my models are?
    * 僅談論CNN model，我認未如果是設想情況將camera放在一片森林裡(架高一點)，我的model從只有煙霧到真的發生火災都可以很快的預測到，可以看result影片。
    * 目前想到的缺點可能就是不能放在楓葉林裡吧，因為我的train data主要是拿正常的綠色森林去作比對，但楓葉林這種感覺就要特別作另一個model去預測可能才會比較準，所以就一般森林而言我的model算滿實用的。 

https://hackmd.io/GvUekqfbQ56OZ9l-FbRoUw#