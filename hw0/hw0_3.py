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

# 計算每個欄位之min, max
max_global_active_power = parsed_data.map(lambda x: x[0]).reduce(max)
min_global_active_power = parsed_data.map(lambda x: x[0]).reduce(min)

max_global_reactive_power = parsed_data.map(lambda x: x[1]).reduce(max)
min_global_reactive_power = parsed_data.map(lambda x: x[1]).reduce(min)

max_voltage = parsed_data.map(lambda x: x[2]).reduce(max)
min_voltage = parsed_data.map(lambda x: x[2]).reduce(min)

max_global_intensity = parsed_data.map(lambda x: x[3]).reduce(max)
min_global_intensity = parsed_data.map(lambda x: x[3]).reduce(min)

# 根據公式，進行MinMaxNormalization
normalized_data = parsed_data.map(lambda x:  ((x[0] - min_global_active_power) / (max_global_active_power - min_global_active_power),
                                              (x[1] - min_global_reactive_power) / (max_global_reactive_power - min_global_reactive_power),
                                              (x[2] - min_voltage) / (max_voltage - min_voltage),
                                              (x[3] - min_global_intensity) / (max_global_intensity - min_global_intensity)))

# 將分區計算的結果匯入為一個txt文件中
normalized_data.coalesce(1).saveAsTextFile("path_to_output.txt")

sc.stop()