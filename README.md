Reading data
Importing the data
my = readtable('weather_train.csv');
Data Preprocessing
Removing missing values and droping non relevent columns
final_table= removevars(my,{'Var1'});
final_table=final_table(~any(ismissing(final_table),2),:);
Detect and remove outliers
figure;
boxplot([final_table.Var10,final_table.Var9,final_table.Var7],'Notch','on','Labels',{'Wind speed','Relative Humidity','Temperature'});
title('Detect outliers Based on Wind Speed, Relative Humidity and Temperature');

data = final_table.Var12;
threshold = 3 * std( data );
validRange = mean( data ) + [-1 1] * threshold;
[m,n] = size(final_table);
for i = i:m
    if data >= validRange(1) & data <= validRange(2)
        final_table(i,:) = [];
    end
end
Spliting data as feature and target
X = table2array(final_table(:,1:n-1));
y = table2array(final_table(:,end));
Normization or Scaling
HIstogram visualization
histogram(y,60)
title('Before Normalization of target data')
Normalize the Target and transform the output
y2 = log(1+y)
histogram(y2,60)
title('After Normalizing the target data')
Correlation between solar power and temperature
plot(X(:,7),y,'o')
title('Corelation of Solar power and temperature');
xlabel('Temperature')
ylabel('Solar power')
Normalizing feature inputs
for i = 1:n-1
    X2(:,i) = (X(:,i) - min(X(:,i)))/max(X(:,i)-min(X(:,i)));   
end
histogram(X2(:,11),10)
title('Normalized feature Input data histogram')

Artificial Neural Network
Model creation
xt = X2';
yt = y2';
hiddenLayerSize = 30;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 30/100;
net.divideParam.testRatio = 0/100;
[net,tr] = train(net,xt,yt)
Performance
yTrain = exp(net(xt(:,tr.trainInd)))-1;
yTrainTrue = exp(yt(tr.trainInd))-1;

yVal = exp(net(xt(:,tr.valInd)))-1;
yValTrue = exp(yt(tr.valInd))-1;

Rmse_train = sqrt(mean(yTrain-yTrainTrue).^2)
Rmse_val = sqrt(mean(yVal-yValTrue).^2)
Visualize the prediction from ANN model
plot(yTrainTrue,yTrain,'x')
xlabel('Predicted output')
ylabel('Actual output')
title('Predicted vs Actual in regression')

Optimize number of neuron of hidden size
for i=1:60
    hiddenLayerSize = i;
    net = fitnet(hiddenLayerSize);
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 30/100;
    net.divideParam.testRatio = 0/100;
    [net,tr] = train(net,xt,yt)
    
    yTrain = exp(net(xt(:,tr.trainInd)))-1;
    yTrainTrue = exp(yt(tr.trainInd))-1;
    
    yVal = exp(net(xt(:,tr.valInd)))-1;
    yValTrue = exp(yt(tr.valInd))-1;
    
    Rmse_train(i) = sqrt(mean(yTrain-yTrainTrue).^2)
    Rmse_val(i) = sqrt(mean(yVal-yValTrue).^2)
    
end
Selecting optimal number of neuron in hidden layer 
plot(1:60,Rmse_train);hold on;
plot(1:60, Rmse_val);hold off;

Custom test data prediction
Predict with Indian csv files
mytest = readtable('India Dataset.csv');
test= removevars(mytest,{'Var1'});
test=test(~any(ismissing(test),2),:);
test = table2array(test(:,1:n-1));

% y = table2array(test(:,end));

[m,n] = size(test)

for i = 1:n
    test(:,i) = (test(:,i) - min(test(:,i)))/max(test(:,i)-min(test(:,i)));   
end

xtest = test';
for i = 1:m
    Indian_output(i,1) = exp(net(xt(:,i)))-1;
end
plot(1:600,Indian_output(1:600,1),'o')
xlabel('No of Days')
ylabel('Predicted Solar power from features')
title('Predicted solar power in Indian dataset')


Predict with India Dataset.xlsx files
mytest = readtable('MALASYIA 2.csv');
test= removevars(mytest,{'Var1'});
test=test(~any(ismissing(test),2),:);
test = table2array(test(:,1:n-1));

% y = table2array(test(:,end));

[m,n] = size(test)

for i = 1:n
    test(:,i) = (test(:,i) - min(test(:,i)))/max(test(:,i)-min(test(:,i)));   
end

xtest = test';
for i = 1:m
    Malasyia_output(i,1) = exp(net(xt(:,i)))-1;
end
plot(1:754,Malasyia_output,'o')
xlabel('No of Days')
ylabel('Predicted Solar power from features')
title('Predicted solar power in Malasyian dataset')

