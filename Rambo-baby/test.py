import re
matches = []
categories_to_retain = ['SOLID',
     'GEOMETRIC',
     'FLORAL',
     'BOTANICAL',
     'STRIPES',
     'ABSTRACT',
     'ANIMAL',
     'GRAPHIC PRINT',
     'ORIENTAL',
     'DAMASK',
     'TEXT',
     'CHEVRON',
     'PLAID',
     'PAISLEY',
     'SPORTS']

x = " Beautiful Art By  Design Studio **graphic print** Creates A **TEXT** Design For This Art Driven Duvet. Printed In Remarkable Detail On A Woven Duvet, This Is An Instant Focal Point Of Any Bedroom. The Fabric Is Woven Of Easy Care Polyester And Backed With A Soft Poly/Cotton Blend Fabric. The Texture In The Fabric Gives Dimension And A Unique Look And Feel To The Duvet."

x = x.upper()

print(x)

def searchWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

for cat in categories_to_retain:
    return_value = searchWholeWord(cat)(x)
    if return_value:
        matches.append(cat)

print(matches)