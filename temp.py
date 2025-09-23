print(sk.root)

print(sk.child_nodes(sk.root))

print(sk.child_nodes(3027))
print(sk_dict['compartment'][3027])
sk_dict

# print(sk.compartment[sk.root])

#gets the position.

print(sk_dict['compartment'][sk.root])

print(sk_dict['radius'][sk.root])


# print(sk.cover_paths)


#go through each segment (that is axonal)

print("branch pts: ", sk.branch_points)

print("segments: ", sk.segments[45])

pt5 = 1960

print(sk.child_nodes(pt5))
print(sk.parent_nodes(pt5))

p1 = 1844
p2 = 1882

print(sk.vertices[p1])
print(sk.vertices[p2])

print(np.linalg.norm([sk.vertices[p1] - sk.vertices[p2]]))


sk_dict['radius'][sk.segments[10][0]]


