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

# 計算平均值
"""
用total除以count

total = parsed_data.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]))
count = parsed_data.count()

average_global_active_power = total[0] / count
average_global_reactive_power = total[1] / count
average_voltage = total[2] / count
average_global_intensity = total[3] / count
"""

mean_global_active_power = parsed_data.map(lambda x: x[0]).mean()
mean_global_reactive_power = parsed_data.map(lambda x: x[1]).mean()
mean_voltage = parsed_data.map(lambda x: x[2]).mean()
mean_global_intensity = parsed_data.map(lambda x: x[3]).mean()

# 列印平均值結果
"""
將結果分開列印
print("Mean Global Active Power:", mean_global_active_power)
print("Mean Global Reactive Power:", mean_global_reactive_power)
print("Mean Voltage:", mean_voltage)
print("Mean Global Intensity:", mean_global_intensity)
"""

print("Mean values:", (mean_global_active_power, mean_global_reactive_power, mean_voltage, mean_global_intensity))

# 計算標準差
std_global_active_power = parsed_data.map(lambda x: x[0]).stdev()
std_global_reactive_power = parsed_data.map(lambda x: x[1]).stdev()
std_voltage = parsed_data.map(lambda x: x[2]).stdev()
std_global_intensity = parsed_data.map(lambda x: x[3]).stdev()


# 列印標準差結果
print("STD values:", (std_global_active_power, std_global_reactive_power, std_voltage, std_global_intensity))

#sc.stop()