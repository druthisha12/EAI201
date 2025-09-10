import json
import re
from datetime import datetime

class BotBrain:
    def __init__(self, campus_graph, search_algorithms):
        """
        Initialize the BotBrain chatbot
        
        Args:
            campus_graph: Dictionary representing the campus graph with nodes and edges
            search_algorithms: Dictionary containing the search algorithm functions
        """
        self.campus_graph = campus_graph
        self.algorithms = search_algorithms
        self.current_algorithm = 'A*'  # Default algorithm
        
        # Building information database
        self.building_info = {
            "academic block a": {
                "description": "Academic Block A",
                "services": ["Classrooms", "Faculty Offices", "Computer Lab"],
                "hours": "7:00 AM - 7:00 PM"
            },
            "academic block b": {
                "description": "Academic Block B",
                "services": ["Classrooms", "Science Labs", "Conference Room"],
                "hours": "7:00 AM - 7:00 PM"
            },
            "academic block c": {
                "description": "Academic Block C",
                "services": ["Classrooms", "Engineering Labs", "Workshop"],
                "hours": "7:00 AM - 7:00 PM"
            },
            "library": {
                "description": "University Library",
                "services": ["Study Halls", "Research Section", "Digital Resources"],
                "hours": "8:00 AM - 10:00 PM"
            },
            "admin building": {
                "description": "Administration Building",
                "services": ["Registrar Office", "Fee Payment Counter", "Student Affairs"],
                "hours": "9:00 AM - 5:00 PM"
            },
            "main hostel": {
                "description": "Main Hostel",
                "services": ["Student Accommodation", "Common Room", "Mess"],
                "hours": "24/7 with curfew"
            },
            "canteen": {
                "description": "University Canteen",
                "services": ["Food Court", "Snacks Counter", "Beverages"],
                "hours": "8:00 AM - 8:00 PM"
            },
            "sports complex": {
                "description": "Sports Complex",
                "services": ["Gymnasium", "Basketball Court", "Swimming Pool"],
                "hours": "6:00 AM - 9:00 PM"
            },
            "medical center": {
                "description": "Medical Center",
                "services": ["First Aid", "Doctor Consultation", "Pharmacy"],
                "hours": "9:00 AM - 5:00 PM (Emergency: 24/7)"
            },
            "main gate": {
                "description": "Main Gate",
                "services": ["Security Office", "Visitor Registration"],
                "hours": "24/7"
            },
            "student center": {
                "description": "Student Center",
                "services": ["Student Lounge", "Club Offices", "Cafeteria"],
                "hours": "8:00 AM - 10:00 PM"
            },
            "auditorium": {
                "description": "University Auditorium",
                "services": ["Events", "Conferences", "Performances"],
                "hours": "As per scheduled events"
            }
        }
        
        # Common user queries and patterns
        self.patterns = {
            'greeting': [r'hello', r'hi', r'hey', r'greetings'],
            'farewell': [r'bye', r'goodbye', r'see you', r'quit', r'exit'],
            'thanks': [r'thank', r'appreciate', r'grateful'],
            'find_path': [r'path from (.*) to (.*)', r'route from (.*) to (.*)', 
                         r'navigate from (.*) to (.*)', r'how to get to (.*) from (.*)',
                         r'directions to (.*)'],
            'building_info': [r'information about (.*)', r'tell me about (.*)', 
                             r'what is (.*)', r'services at (.*)'],
            'change_algorithm': [r'use (.*) algorithm', r'switch to (.*)', 
                                r'change to (.*)', r'set algorithm to (.*)'],
            'algorithm_info': [r'which algorithm', r'current algorithm', 
                              r'what algorithm are you using'],
            'list_buildings': [r'list buildings', r'what locations', 
                              r'available places', r'where can I go']
        }
        
        print("BotBrain initialized! How can I help you navigate the campus today?")
        print("Type 'help' to see available commands.")
    
    def process_query(self, query):
        """
        Process the user's query and return an appropriate response
        """
        query_lower = query.lower().strip()
        
        # Check for greeting
        if any(re.match(pattern, query_lower) for pattern in self.patterns['greeting']):
            return "Hello! I'm BotBrain, your campus navigation assistant. How can I help you today?"
        
        # Check for farewell
        if any(re.match(pattern, query_lower) for pattern in self.patterns['farewell']):
            return "Goodbye! Have a great day on campus!"
        
        # Check for thanks
        if any(re.match(pattern, query_lower) for pattern in self.patterns['thanks']):
            return "You're welcome! Is there anything else I can help you with?"
        
        # Check for help
        if query_lower == 'help':
            return self.get_help_message()
        
        # Check for path finding
        for pattern in self.patterns['find_path']:
            match = re.search(pattern, query_lower)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    source, destination = groups
                elif len(groups) == 1:
                    # Handle "directions to X" pattern
                    destination = groups[0]
                    # In a real implementation, you might ask for source
                    return f"I'd be happy to help you get to {destination}. Please tell me where you're starting from."
                else:
                    continue
                
                source = self.normalize_location_name(source.strip())
                destination = self.normalize_location_name(destination.strip())
                
                if source not in self.campus_graph['nodes']:
                    return f"Sorry, I don't recognize '{source}'. Please check the building name."
                if destination not in self.campus_graph['nodes']:
                    return f"Sorry, I don't recognize '{destination}'. Please check the building name."
                
                return self.find_path(source, destination)
        
        # Check for building information
        for pattern in self.patterns['building_info']:
            match = re.search(pattern, query_lower)
            if match:
                building = self.normalize_location_name(match.group(1).strip())
                if building in self.building_info:
                    return self.get_building_info(building)
                else:
                    return f"Sorry, I don't have information about {building}. Please check the building name."
        
        # Check for algorithm change
        for pattern in self.patterns['change_algorithm']:
            match = re.search(pattern, query_lower)
            if match:
                algorithm = match.group(1).strip().upper()
                if algorithm in ['BFS', 'DFS', 'UCS', 'A*', 'A STAR']:
                    if algorithm == 'A STAR':
                        algorithm = 'A*'
                    self.current_algorithm = algorithm
                    return f"Switched to {algorithm} algorithm for path finding."
                else:
                    return f"Sorry, I don't recognize the {algorithm} algorithm. Available algorithms: BFS, DFS, UCS, A*."
        
        # Check for algorithm info
        for pattern in self.patterns['algorithm_info']:
            if re.search(pattern, query_lower):
                return f"I'm currently using the {self.current_algorithm} algorithm for path finding."
        
        # Check for list buildings
        for pattern in self.patterns['list_buildings']:
            if re.search(pattern, query_lower):
                buildings = list(self.campus_graph['nodes'].keys())
                return "Available buildings: " + ", ".join(buildings)
        
        # Default response for unrecognized queries
        return "I'm not sure I understand. Try asking for directions between buildings or information about a specific location. Type 'help' for more options."
    
    def normalize_location_name(self, name):
        """
        Normalize building names to match our graph nodes
        """
        name = name.lower()
        mappings = {
            'academic a': 'academic block a',
            'academic b': 'academic block b', 
            'academic c': 'academic block c',
            'academic block a': 'academic block a',
            'academic block b': 'academic block b',
            'academic block c': 'academic block c',
            'admin': 'admin building',
            'admin block': 'admin building',
            'hostel': 'main hostel',
            'medical': 'medical center',
            'gate': 'main gate',
            'student centre': 'student center'
        }
        
        return mappings.get(name, name)
    
    def find_path(self, source, destination):
        """
        Find a path between two locations using the selected algorithm
        """
        try:
            if self.current_algorithm not in self.algorithms:
                return f"Error: {self.current_algorithm} algorithm not available."
            
            # Call the appropriate search algorithm
            path, distance, nodes_explored = self.algorithms[self.current_algorithm](source, destination)
            
            if not path:
                return f"Sorry, I couldn't find a path from {source} to {destination}."
            
            # Calculate estimated walking time (assuming 1.4 m/s walking speed)
            walking_time_minutes = int((distance / 1.4) / 60)
            
            response = f"Path from {source} to {destination} using {self.current_algorithm}:\n"
            response += f"Route: {' -> '.join(path)}\n"
            response += f"Total distance: {distance} meters\n"
            response += f"Estimated walking time: {walking_time_minutes} minutes\n"
            response += f"Nodes explored: {nodes_explored}"
            
            return response
            
        except Exception as e:
            return f"Error finding path: {str(e)}"
    
    def get_building_info(self, building):
        """
        Get information about a building
        """
        info = self.building_info[building]
        response = f"{info['description']}:\n"
        response += f"Services: {', '.join(info['services'])}\n"
        response += f"Hours: {info['hours']}"
        
        return response
    
    def get_help_message(self):
        """
        Return help message with available commands
        """
        help_text = """
I can help you with:
- Finding paths between buildings (e.g., "path from library to canteen")
- Providing information about buildings (e.g., "tell me about the library")
- Changing search algorithms (e.g., "use UCS algorithm")
- Listing available buildings (e.g., "list buildings")

Available algorithms: BFS, DFS, UCS, A*

You can also say hello, goodbye, or thanks!
"""
        return help_text

# Example usage and test function
def test_botbrain():
    """
    Test function to demonstrate the BotBrain chatbot
    """
    # Mock campus graph (replace with your actual graph)
    campus_graph = {
        'nodes': {
            'academic block a': {'x': 0, 'y': 0},
            'academic block b': {'x': 100, 'y': 0},
            'academic block c': {'x': 200, 'y': 0},
            'library': {'x': 0, 'y': 100},
            'admin building': {'x': 100, 'y': 100},
            'main hostel': {'x': 200, 'y': 100},
            'canteen': {'x': 0, 'y': 200},
            'sports complex': {'x': 100, 'y': 200},
            'medical center': {'x': 200, 'y': 200},
            'main gate': {'x': 100, 'y': 300},
            'student center': {'x': 200, 'y': 300},
            'auditorium': {'x': 0, 'y': 300}
        },
        'edges': {
            'academic block a': [('academic block b', 100), ('library', 150)],
            'academic block b': [('academic block a', 100), ('academic block c', 100), ('admin building', 100)],
            'academic block c': [('academic block b', 100), ('main hostel', 150)],
            'library': [('academic block a', 150), ('admin building', 100), ('canteen', 150)],
            'admin building': [('academic block b', 100), ('library', 100), ('sports complex', 150)],
            'main hostel': [('academic block c', 150), ('sports complex', 100), ('medical center', 100)],
            'canteen': [('library', 150), ('sports complex', 100), ('auditorium', 150)],
            'sports complex': [('admin building', 150), ('main hostel', 100), ('canteen', 100), ('main gate', 150)],
            'medical center': [('main hostel', 100), ('student center', 150)],
            'main gate': [('sports complex', 150), ('student center', 100)],
            'student center': [('medical center', 150), ('main gate', 100)],
            'auditorium': [('canteen', 150), ('main gate', 200)]
        }
    }
    
    # Mock search algorithms (replace with your actual implementations)
    def bfs_search(source, destination):
        # Simplified mock implementation
        path = [source, "intermediate point", destination]
        distance = 250
        nodes_explored = 5
        return path, distance, nodes_explored
    
    def dfs_search(source, destination):
        # Simplified mock implementation
        path = [source, "different intermediate point", destination]
        distance = 300
        nodes_explored = 7
        return path, distance, nodes_explored
    
    def ucs_search(source, destination):
        # Simplified mock implementation
        path = [source, "optimal intermediate point", destination]
        distance = 200
        nodes_explored = 8
        return path, distance, nodes_explored
    
    def a_star_search(source, destination):
        # Simplified mock implementation
        path = [source, "best intermediate point", destination]
        distance = 180
        nodes_explored = 4
        return path, distance, nodes_explored
    
    search_algorithms = {
        'BFS': bfs_search,
        'DFS': dfs_search,
        'UCS': ucs_search,
        'A*': a_star_search
    }
    
    # Initialize the bot
    bot = BotBrain(campus_graph, search_algorithms)
    
    # Test queries
    test_queries = [
        "hello",
        "path from library to canteen",
        "tell me about the library",
        "use UCS algorithm",
        "which algorithm are you using?",
        "list buildings",
        "thank you",
        "bye"
    ]
    
    print("\n=== Testing BotBrain ===")
    for query in test_queries:
        print(f"\nUser: {query}")
        response = bot.process_query(query)
        print(f"BotBrain: {response}")
    
    # Interactive mode
    print("\n=== Interactive Mode ===")
    print("Type your queries (or 'quit' to exit):")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("BotBrain: Goodbye!")
                break
            
            response = bot.process_query(user_input)
            print(f"BotBrain: {response}")
            
        except KeyboardInterrupt:
            print("\nBotBrain: Goodbye!")
            break
        except Exception as e:
            print(f"BotBrain: Sorry, I encountered an error: {str(e)}")

if __name__ == "__main__":
    test_botbrain()