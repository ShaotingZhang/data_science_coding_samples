import zen
import urllib
import time
import unicodedata


def add_edge(names,G):    #  add edge into graph
  for i in range(len(names)-1):
    for j in range(i+1,len(names)):
      if G.has_edge(names[i],names[j]):
#        print 'exist'
        eidx = G.edge_idx(names[i],names[j])
        G.set_weight_(eidx,G.weight_(eidx)+1)
      else:
#        print 'new'
        G.add_edge(names[i],names[j],weight=1)

def edit_name(names,G):   # rewrite the name to compare nodes
  for i in range(len(names)):
    d = names[i].find('(')
    if d > 1:
      names[i] = names[i][:d-1]
  names = list(set(names))

  temp = []
  for name in names:    
    space = name.find(' ')
    if space < 0:
      name = name[0].upper() + name[1:].lower()
    else:
      name = name[0].upper() + name[1:space].lower()
    temp.append(name)
  names = temp

  for i in range(len(names)):    # add nodes to graph
    if names[i] == 'High':
      names[i] = 'Sparrow'
    if names[i] not in G:
      G.add_node(names[i])

#  print names
  return names

def GenerateGraph(filename,G):     #  for each file to generate the graph
  f = open(filename)
  lines = f.readlines()
  f.close
  names = []
  for line in lines:
    if line[0] == '-':
      names = edit_name(names,G)
      if len(names) > 1:
        add_edge(names,G)
      names = []
  
    p = line.find(':')
    if p >= 0:
      name = line[:p]
      if name != 'CUT TO':
        if name.find('\xa1') < 0:
          names.append(name)
  return G

def main():
  address = []
  address.append('http://gameofthrones.wikia.com/wiki/Category:Characters?display=page&sort=mostvisited')
  #address.append('http://gameofthrones.wikia.com/wiki/Category:Characters?display=page&sort=mostvisited&pagefrom=Forel%2C+Syrio%0ASyrio+Forel#mw-pages')
  #address.append('http://gameofthrones.wikia.com/wiki/Category:Characters?display=page&sort=mostvisited&pagefrom=Lannister%2C+Tommen+II%0ATommen+II+Lannister#mw-pages')
  #address.append('http://gameofthrones.wikia.com/wiki/Category:Characters?display=page&sort=mostvisited&pagefrom=Selmy%2C+Barristan%0ABarristan+Selmy#mw-pages')
  #address.append('http://gameofthrones.wikia.com/wiki/Category:Characters?display=page&sort=mostvisited&pagefrom=White+Walker+%28Winter+is+Coming%29#mw-pages')

  PAGE_START = 'mw-pages'
  PAGE_END = 'class=\"bodyAd'

  CATEGORY_BREAK = '/wiki/'
  CATEGORY_URL = 'http://gameofthrones.wikia.com/wiki/'

  G = zen.Graph()
  nodes = []
  #index = [2,3,3,2]
  index = [0,0,0,0]
  for i in range(1):
    url = address[i]
    f = urllib.urlopen(url)
    s = f.read()
    f.close()

    contents = s.split(PAGE_START)[index[i]] # keep only the part after the first occurance
    contents = contents.split(PAGE_END)[0]
    hrefs = contents.split('<a')[1:]

    for href in hrefs:
      href = href.split('</a>')[0]
      if href.find(CATEGORY_BREAK) >= 0:
        caturl = href.split(CATEGORY_BREAK)[1]
        temp1 = caturl.find('title')
        if temp1 != -1:
          caturl = caturl[:temp1-2]
          nodes.append(caturl)

  #  print len(nodes)
  #print nodes

  #for i in range(len(nodes)):  # add nodes to graph
  #  node = nodes[i]
  #  if node not in G:
  #    G.add_node(node)

  for i in range(1,7):
      filename = 'script' + str(i) + '.txt'
      G = GenerateGraph(filename,G)

  listnames = []
  for i in range(G.num_nodes):
      listnames.append(G.node_object(i))
  #print listnames
  #print G.num_nodes
  #print G.num_edges

  zen.io.gml.write(G,'GameOfThrones.gml')

if __name__ == '__main__':
  main()
