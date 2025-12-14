try:
    with open('final_benchmark_51runs.txt', 'r', encoding='utf-16') as f:
        content = f.read()
except:
    with open('final_benchmark_51runs.txt', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

if "FINAL BENCHMARK RESULTS" in content:
    print(content.split("FINAL BENCHMARK RESULTS")[1])
else:
    print("Table not found in file.")
