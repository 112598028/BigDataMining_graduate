from pyspark import SparkConf, SparkContext

# 讀取檔案
power_data_file = "./hw0/household_power_consumption.txt"

# 建立Spark Context
conf = SparkConf().setAppName("TextProcess")
sc = SparkContext(conf=conf)

text_file = sc.textFile(power_data_file)

# 定義選取欄位之function
def parse_line(line):
    fields = line.split(";")
    try:
        global_active_power = float(fields[2])
        global_reactive_power = float(fields[3])
        voltage = float(fields[4])
        global_intensity = float(fields[5])
        return (global_active_power, global_reactive_power, voltage, global_intensity)
    
    # 無法解析時，將整行資料做-1.0處理
    except:
        return -1.0

# 使用map進行解析
parsed_data = text_file.map(parse_line).filter(lambda x: x!=-1.0)

# 使用reduce計算max, min, count
min_values = parsed_data.reduce(lambda x, y: tuple(min(a, b) for a, b in zip(x, y)))
max_values = parsed_data.reduce(lambda x, y: tuple(max(a, b) for a, b in zip(x, y)))
count = parsed_data.count()

# 第一題：顯示結果
print("Minimum values:", min_values)
print("Maximum values:", max_values)
# 因為資歷整行做統一的-1.0處理，因此每個column的資料筆數將會相同
print("Count:", count) 

#sc.stop()