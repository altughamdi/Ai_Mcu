% clc, clear all; close all;
load fisheriris;
dataset=meas; 
res= species;
result=zeros(150,1);


for i=1:1:150
    if(res(i)=="versicolor")
        result(i)=1;
    end
    if(res(i)=="virginica")
        result(i)=2;
    end
end

csvwrite('0001.csv',dataset);
csvwrite('0002.csv',result);
