def divide_numbers(a, b):
    return a / b  # Possible division by zero error

def main():
    numbers = [10, 5, 0, 2]
    
    for i in range(len(numbers)):
        print(f"Result: {divide_numbers(100, numbers[i])}")  # Error when numbers[i] is 0

if __name__ == "__main__":
    main()
