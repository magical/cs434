
import pca

data = pca.read_data()
ev, el = pca.pca(data, 10)
#print(map(list, ev))
#print(el)
for x in el:
    if x.imag == 0:
        print(x.real)
    else:
        print(x)

