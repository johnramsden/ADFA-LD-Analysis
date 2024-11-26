def main():

    # Read the input file containing system calls and their numbers
    input_file = "mapped_system_calls.csv"  # Replace with your file name
    output_file = "bpf_ksyscall_output.c"  # Output file to write the results

    # Open the input file and process each line
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            # Split the line into number and syscall name
            parts = line.strip().split()
            if len(parts) != 2:
                continue  # Skip lines that do not match the expected format

            number, syscall = parts[0], parts[1]

            # Generate the BPF program text
            program_text = f'''
    SEC("ksyscall/{syscall}")
    int BPF_KSYSCALL({syscall}_entry)
    {{
        bpf_printk("{number}");
        return 0;
    }}
    '''

            # Write to the output file
            outfile.write(program_text)

    print(f"Generated BPF programs have been written to {output_file}.")

if __name__ == "__main__":
    main()
