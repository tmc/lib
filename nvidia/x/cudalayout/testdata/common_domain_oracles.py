from pycute import Layout, greatest_common_domain
pairs=[((4,6,4),(3,3)),((10,15,10),((3,5),10)),((2,3,5),(5,3,2)),((1,4,6,4),((3,1),3)),((2,6),(3,4)),((4,3),(2,6)),((2,5,3),(5,2,3))]
for a,b in pairs:
 print(a,b,greatest_common_domain(Layout(a),Layout(b)))
