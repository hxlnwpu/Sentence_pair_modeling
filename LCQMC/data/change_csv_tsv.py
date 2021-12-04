# with open('dev.csv','r',encoding='utf-8') as f:
#     data = f.read().replace(',','\t')
# with open('dev.tsv','w',encoding='utf-8') as f:
#     f.write(data)
# with open('test.csv','r',encoding='utf-8') as f:
#     data = f.read().replace(',','\t')
# with open('test.tsv','w',encoding='utf-8') as f:
#     f.write(data)
# with open('train.csv','r',encoding='utf-8') as f:
#     data = f.read().replace(',','\t')
# with open('train.tsv','w',encoding='utf-8') as f:
#     f.write(data)
with open('test.tsv','r',encoding='utf-8') as f:
    data = f.read().replace(',','，').replace('\t',',')
with open('test.csv','w',encoding='utf-8') as f:
    f.write(data)
with open('train.tsv','r',encoding='utf-8') as f:
    data = f.read().replace(',','，').replace('\t',',')
with open('train.csv','w',encoding='utf-8') as f:
    f.write(data)