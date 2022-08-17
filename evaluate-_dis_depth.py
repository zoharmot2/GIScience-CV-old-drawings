import csv


# read data from the features file
def read_points_data(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        recs = []
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                names = row
            else:
                recs.append(row)
            line_count += 1
        print(f'Processed {line_count} lines.')
        return recs



# open the features file
file  = 'C:/Users/zoharmot/Dropbox (University of Haifa)/m-academic/research/3c_georef_drawings_ML/python/features.csv'

recs = read_points_data(file)
res = [8]
count = 1
for rec in recs:
    res[0] = str(count)
    res[1] = rec['Symbol']
    for rec2 in recs:
        if rec['Symbol' != rec2['Symbol']]:
            res[3] = rec2['Symbol']









# create new file