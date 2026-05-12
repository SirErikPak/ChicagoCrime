# Define mapping for time of day based on Hour of Day
time_of_day_mapping = {
        'Early Morning': '04AM - 07AM',
        'Morning': '08AM - 11AM',
        'Afternoon': '12PM - 03PM',
        'Evening': '04PM - 07PM',
        'Night': '08PM - 11PM',
        'Late Night': '12AM - 03AM',
        }

# Define FBI Codes mapping
fbi_codes = {
        "01A": {"desc": "Homicide - 1st or 2nd Degree", "is_index": True},
        "01B": {"desc": "Involuntary Manslaughter / Reckless Homicide", "is_index": True},
        "02":  {"desc": "Criminal Sexual Assault", "is_index": True},
        "03":  {"desc": "Robbery", "is_index": True},
        "04A": {"desc": "Aggravated Assault", "is_index": True},
        "04B": {"desc": "Aggravated Battery", "is_index": True},
        "05":  {"desc": "Burglary", "is_index": True},
        "06":  {"desc": "Larceny - Theft", "is_index": True},
        "07":  {"desc": "Motor Vehicle Theft", "is_index": True},
        "09":  {"desc": "Arson", "is_index": True},
        "08A": {"desc": "Simple Assault", "is_index": False},
        "08B": {"desc": "Simple Battery", "is_index": False},
        "10":  {"desc": "Forgery and Counterfeiting", "is_index": False},
        "11":  {"desc": "Fraud", "is_index": False},
        "12":  {"desc": "Embezzlement", "is_index": False},
        "13":  {"desc": "Stolen Property (Buy, Receive, Possess)", "is_index": False},
        "14":  {"desc": "Vandalism", "is_index": False},
        "15":  {"desc": "Weapons Violations", "is_index": False},
        "16":  {"desc": "Prostitution", "is_index": False},
        "17":  {"desc": "Sex Offense - Criminal Sexual Abuse", "is_index": False},
        "18":  {"desc": "Drug Abuse Violations", "is_index": False},
        "19":  {"desc": "Gambling", "is_index": False},
        "20":  {"desc": "Offenses Against Family and Children", "is_index": False},
        "22":  {"desc": "Liquor Laws", "is_index": False},
        "24":  {"desc": "Disorderly Conduct", "is_index": False},
        "26":  {"desc": "Miscellaneous Non-Index Offenses", "is_index": False}
    }


# Mapping of Chicago Community Area ID to Official Neighborhood Name
chicago_communities = {
        "01": "Rogers Park", "02": "West Ridge", "03": "Uptown", "04": "Lincoln Square", "05": "North Center",
        "06": "Lake View", "07": "Lincoln Park", "08": "Near North Side", "09": "Edison Park", "10": "Norwood Park",
        "11": "Jefferson Park", "12": "Forest Glen", "13": "North Park", "14": "Albany Park", "15": "Portage Park",
        "16": "Irving Park", "17": "Dunning", "18": "Mont Clare", "19": "Belmont Cragin", "20": "Hermosa",
        "21": "Avondale", "22": "Logan Square", "23": "Humboldt Park", "24": "West Town", "25": "Austin",
        "26": "West Garfield Park", "27": "East Garfield Park", "28": "Near West Side", "29": "North Lawndale", "30": "South Lawndale",
        "31": "Lower West Side", "32": "Loop", "33": "Near South Side", "34": "Armour Square", "35": "Douglas",
        "36": "Oakland", "37": "Fuller Park", "38": "Grand Boulevard", "39": "Kenwood", "40": "Washington Park",
        "41": "Hyde Park", "42": "Woodlawn", "43": "South Shore", "44": "Chatham", "45": "Avalon Park",
        "46": "South Chicago", "47": "Burnside", "48": "Calumet Heights", "49": "Roseland", "50": "Pullman",
        "51": "South Deering", "52": "East Side", "53": "West Pullman", "54": "Riverdale", "55": "Hegewisch",
        "56": "Garfield Ridge", "57": "Archer Heights", "58": "Brighton Park", "59": "McKinley Park", "60": "Bridgeport",
        "61": "New City", "62": "West Elsdon", "63": "Gage Park", "64": "Clearing", "65": "West Lawn",
        "66": "Chicago Lawn", "67": "West Englewood", "68": "Englewood", "69": "Greater Grand Crossing", "70": "Ashburn",
        "71": "Auburn Gresham", "72": "Beverly", "73": "Washington Heights", "74": "Mount Greenwood", "105": "Morgan Park",
        "76": "Ohare", "77": "Edgewater"
    }

# Police District Location Mapping: https://www.chicagopolice.org/police-districts/
cpd_districts = {
        "001": "Central",
        "002": "Wentworth",
        "003": "Grand Crossing",
        "004": "South Chicago",
        "005": "Calumet",
        "006": "Gresham",
        "007": "Englewood",
        "008": "Chicago Lawn",
        "009": "Deering",
        "010": "Ogden",
        "011": "Harrison",
        "012": "Near West",
        "014": "Shakespeare",
        "015": "Austin",
        "016": "Jefferson Park",
        "017": "Albany Park",
        "018": "Near North",
        "019": "Town Hall",
        "020": "Lincoln",
        "022": "Morgan Park",
        "024": "Rogers Park",
        "025": "Grand Central"
    }

# Mapping sector represented as Area according to: https://www.chicagopolice.org/statistics-data/crime-statistics/
cpd_sector = {
        '1': ['002', '003', '007', '008', '009'],
        '2': ['004', '005', '006', '022'],
        '3': ['001', '012', '018', '019', '020', '024'],
        '4': ['010', '011', '015'],
        '5': ['014', '016', '017', '025']
    } 