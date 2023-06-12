lst = [96, 128, 96, 64, 82, 52, 128, 21, 97, 0, 16, 87, 96, 0, 128, 253, 91, 80, 96, 4, 54, 16, 97, 2, 2, 3, '\n',
       96, 128, 96, 64, 82, 52, 128, 21, 97, 0, 16, 87, 96, 0, 128, 253, 91, 80, 96, 4, 54, 16, 97, 2, 2, 4, '\n',
       96, 128, 96, 64, 82, 52, 128, 21, 97, 0, 16, 87, 96, 0, 128, 253, 91, 80, 96, 4, 54, 16, 97, 2, 2, 5, '\n',
       96, 128, 96, 64, 82, 52, 128, 21, 97, 0, 16, 87, 96, 0, 128, 253, 91, 80, 96, 4, 54, 16, 97, 2, 2, 3, '\n',
       96, 128, 96, 64, 82, 52, 128, 21, 97, 0, 16, 87, 96, 0, 128, 253, 91, 80, 96, 4, 54, 16, 97, 2, 2, 4, '\n',
       96, 128, 96, 64, 82, 52, 128, 21, 97, 0, 16, 87, 96, 0, 128, 253, 91, 80, 96, 4, 54, 16, 97, 2, 2, 5, '\n']

# Remove '\n'
lst = [elem for elem in lst if elem != '\n']
# Replace commas with spaces
lst = [str(elem) for elem in lst]
lst = " ".join(lst)
# Add a starting single quote
lst = "'" + lst
lst = "'" + lst
# Insert a single quote and comma after every 26 integers(26*3-2)
lst = [lst[i:i+76] + "', '" for i in range(1, len(lst), 76)]
# Join the list back into a single string and remove the trailing comma
lst = "".join(lst)[:-3]
# Add a closing single quote

print(lst)