def vacuum_cleaner():
    print("Vacuum Cleaner Simulation")
    print("Commands: start, stop, left, right, dock, exit")
    
    running = False
    
    while True:
        command = input("Enter command: ").strip().lower()
        
        if command == "start":
            if not running:
                running = True
                print("Vacuum cleaner started.")
            else:
                print("Vacuum cleaner is already running.")
        
        elif command == "stop":
            if running:
                running = False
                print("Vacuum cleaner stopped.")
            else:
                print("Vacuum cleaner is not running.")
        
        elif command == "left":
            if running:
                print("Vacuum cleaner turned left.")
            else:
                print("Start the vacuum first.")
        
        elif command == "right":
            if running:
                print("Vacuum cleaner turned right.")
            else:
                print("Start the vacuum first.")
        
        elif command == "dock":
            if running:
                print("Vacuum cleaner is docking...")
                running = False
            else:
                print("Vacuum cleaner docked.")
        
        elif command == "exit":
            print("Exiting simulation.")
            break
        
        else:
            print("Invalid command! Please enter: start, stop, left, right, dock, exit.")

vacuum_cleaner()
