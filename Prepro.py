import os
import json
import csv


directory=os.getcwd()

for filename in os.listdir(directory):
    if filename.endswith(".txt"):

        # the file to be converted
        filename = os.path.join(filename)

        # intermediate and resultant dictionaries
        # intermediate
        dict2 = {}

        # resultant
        dict1 = {}

        # fields in the sample file
        # fields = list(range(0,17))
        fields = ["label", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8","F9", "F10", "F11", "F12", "F13", "F14", "F15",
                  "F16"]


        with open(filename) as fh:
            # loop variable
            i = 0

            # count variable for employee id creation
            l = 1

            for line in fh:

                # reading line by line from the text file
                description = list(line.strip().split(None, 16))

                # for output see below
                #print(description)

                # for automatic creation of id for each employee
                sno = 'emp' + str(l)

                while i < len(fields):
                    # creating dictionary for each employee

                    temp = description[i].split(':')

                    if len(temp) == 2:
                        dict2[fields[i]] = temp[1]

                    else:
                        dict2[fields[i]] = temp[0]

                    """
                    if there is : , sparse the text and put the second element in json

                    if there is no :, put directly
                    """

                    i = i + 1

                # appending the record of each employee to
                # the main dictionary
                dict1[sno] = dict2

                # reset
                i = 0
                dict2 = {}

                l = l + 1

        # creating json file

        if filename.endswith("test.txt"):
            out_file = open("letter_test.json", "w")
            json.dump(dict1, out_file, indent=16)
            out_file.close()

        else:
            out_file = open("letter_train.json","w")
            json.dump(dict1, out_file, indent=16)
            out_file.close()

print("==================complete makeJsonFile=======================")

for filename in os.listdir(directory):

    if filename.endswith(".json"):
        filename = os.path.join(filename)

        with open(filename) as json_file:
            data = json.load(json_file)

            # now we will open a file for writing
            filename=filename[:-4]+'csv'
            data_file = open(filename, 'w')
            print(filename)

            # create the csv writer object
            csv_writer = csv.writer(data_file)

            # for loof for component
            for i in range(len(data)):
                if i == 0:
                  # Writing headers of CSV file
                  letter_data = data['emp1']
                  header = letter_data.keys()
                  csv_writer.writerow(header)
                  #print(header)

                else:
                    index = 'emp' + str(i)
                    letter_data = data[index]
                    csv_writer.writerow(letter_data.values())
                   # print(letter_data)  # dictionary

            data_file.close()

print("==================complete Json to Csv=======================")

