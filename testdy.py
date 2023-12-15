import dynet as dy

m = dy.ParameterCollection()

#dy.renew_cg()

p = m.add_parameters(4)
a = [1,2,3,4]
obj = sum(p[i] * a[i] for i in range(4))

#print(p.value())

dy.renew_cg()

p.set_value([1,2,3,4])
print(p.value())

p.set_value([2,2,3,4])
#dy.renew_cg()
print(obj.value())
obj.backward()
print(p.value())
