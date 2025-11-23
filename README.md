# AI first course creator

AI powered course creator application creates dynamic course lesson plan and content based on the user provided topic with the experience level in that topic for a specified course duration.

## Tech Stack

**LLM Model:** Google Gemini

**Programming Language:** Python, Google ADK


## Run Locally

Clone the project

```bash
  git clone https://github.com/AravindGShanthi/ai-first-learning-app.git
```

Go to the project directory

```bash
  cd ai-first-learning-app
```

Crete virtual environment
```bash
  python venv venv
  source venv/bin/activate
```

Install dependencies
```bash
  pip install
```

Create .env and update google key
```bash
  GOOGLE_API_KEY=
```

Start the server

```bash
  python main.py
```



## Usage/Examples

```bash
Topic of interest (eg. AI developer, Cloud technologies, Python learning): Advance python concepts
Experience level (eg. 1 year, 2 years, 10 years): 4 years
Duration in days (eg. 1, 7, 10, 15, 30): 15
```
Request requested to create lesson plan for **Advance Python concepts** for a duration of 15 days.  AI Multi-agent feedback loop system create lesson plan for 15 days with one topic/concept per day considering the user experience in that specific topic.

HITL (Human In The Loop) agent asks for user feedback to either APPROVE or REJECT with feedback so that the AI agent system includes user feedback, review and refine the lesson plan.

```bash
Approve the lesson plan? (y/n) [y = approve]: Reject
Optional feedback (enter any comments to pass to the refiner agent): add django and flash for server programming

-- This will restart the review and refinement loop to regenerate the lesson plan to include user feedback

Approve the lesson plan? (y/n) [y = approve]: y
Optional feedback (enter any comments to pass to the refiner agent): None

-- Exit the lesson plan generator loop
```

It will present the final lesson plan as a option for user selection.

```bash
Day 1: Advanced Decorators and Metaclasses in Python
Day 2: Descriptors and Custom Attribute Access
Day 3: Asynchronous Programming with `asyncio` (Advanced Patterns & Libraries)
Day 4: Concurrency with `threading` and `multiprocessing` (Advanced Synchronization & IPC)
Day 5: CPython Internals: GIL, Garbage Collection, and Object Model
Day 6: Advanced Type Hinting and Static Analysis (`mypy`, Protocols, Generics)
Day 7: Performance Optimization: Profiling, `Cython`, and `Numba`
Day 8: Flask Framework: Advanced Patterns, Blueprints, and REST API Development
Day 9: Django Framework: Advanced Models, ORM, and REST APIs with DRF
Day 10: Extending Python with C/C++ (`ctypes` and an Introduction to Python C API)
```

You can see it has included Flask and Django framework in the lesson plan based on user feedback.

Choose the days from 1 to 10 for topic selection.
Value -1 for exit

```bash
Please select the concept to learn. Eg. If you want to select day 5 please enter 5: 2
```

It will create the content for the selected topic/concept in JSON format

```bash
content_generator_agent >
{
  "title": "Python Descriptors and Custom Attribute Access",
  "description": "Explore Python's powerful mechanisms for customizing attribute behavior. This lesson covers descriptors, which enable managed attributes with custom logic for getting, setting, or deleting values, and special 'dunder' methods like `__getattr__` and `__getattribute__` for intercepting attribute access.",
  "detailed_explanation": "Python's object model allows for fine-grained control over how attributes are accessed, modified, and deleted. This control is primarily achieved through **descriptors** and custom attribute access methods.\n\n### Descriptors\n\nDescriptors are Python objects that implement one or more of the **descriptor protocol** methods: `__get__`, `__set__`, or `__delete__`. When an instance of a descriptor class is assigned as a class attribute in another class, it becomes a 'managed attribute,' allowing custom behavior upon access.\n\n**Descriptor Protocol Methods:**\n*   `__get__(self, obj, type=None)`: Called to get the attribute's value. `obj` is the instance of the owner class, and `type` is the owner class itself. If accessed directly from the class (e.g., `MyClass.attribute`), `obj` will be `None`.\n*   `__set__(self, obj, value)`: Called to set the attribute's value. `obj` is the instance of the owner class, and `value` is the new value being assigned.\n*   `__delete__(self, obj)`: Called to delete the attribute. `obj` is the instance of the owner class.\n*   `__set_name__(self, owner, name)`: (Python 3.6+) Called automatically when the descriptor is assigned to a class attribute. `owner` is the class where the descriptor is defined, and `name` is the attribute's name. This helps in avoiding hardcoding the attribute name within the descriptor.\n\n**Data vs. Non-Data Descriptors:**\n*   **Data Descriptors:** Implement `__set__` or `__delete__` (and optionally `__get__`). They take precedence over entries in an instance's `__dict__` with the same name.\n*   **Non-Data Descriptors:** Only implement `__get__`. They have lower precedence than entries in an instance's `__dict__`. If an instance has an attribute with the same name, the instance attribute will be returned instead of invoking the non-data descriptor.\n\n**Common Use Cases for Descriptors:**\n*   **Attribute Validation:** Ensuring attributes meet specific type or value constraints.\n*   **Computed Properties:** Lazy loading or dynamically calculating attribute values.\n*   **Logging or Access Control:** Tracking when attributes are accessed or modified.\n*   **ORM Fields:** Abstracting database interactions.\n\n**Example: A Descriptor for Type Validation**\n```python\nclass TypeValidator:\n    def __init__(self, expected_type, name=None):\n        self.expected_type = expected_type\n        self.public_name = name\n\n    def __set_name__(self, owner, name):\n        if self.public_name is None:\n            self.public_name = name # Use the attribute name if not provided\n        self.private_name = '_' + self.public_name # Store value in a private attribute\n\n    def __get__(self, obj, objtype=None):\n        if obj is None:\n            return self # Accessing descriptor from class\n        return obj.__dict__.get(self.private_name, None)\n\n    def __set__(self, obj, value):\n        if not isinstance(value, self.expected_type):\n            raise TypeError(f\"Expected {self.public_name} to be {self.expected_type.__name__}, got {type(value).__name__}\")\n        obj.__dict__[self.private_name] = value\n\nclass Person:\n    name = TypeValidator(str)\n    age = TypeValidator(int)\n\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n\np = Person(\"Alice\", 30)\nprint(f\"Name: {p.name}, Age: {p.age}\") # Output: Name: Alice, Age: 30\n\ntry:\n    p.age = \"thirty\" # Raises TypeError\nexcept TypeError as e:\n    print(e) # Output: Expected age to be int, got str\n```\n\nBuilt-in Python features like `property()`, `classmethod()`, and `staticmethod()` are implemented using descriptors.\n\n### Custom Attribute Access Methods (`__getattr__`, `__getattribute__`, `__setattr__`, `__delattr__`)\n\nThese are special 'dunder' (double underscore) methods that allow classes to customize attribute access at a lower level than descriptors, affecting the entire object's attribute lookup behavior.\n\n*   `__getattr__(self, name)`:\n    *   **Purpose:** This method is called only when an attribute lookup has *not found* the attribute in the usual places (i.e., it's not in the instance's `__dict__` and not in the class's method resolution order, or MRO).\n    *   **Behavior:** It's a fallback mechanism. If defined, it can compute and return the attribute value or raise an `AttributeError`.\n\n*   `__getattribute__(self, name)`:\n    *   **Purpose:** This method is called *unconditionally* for all attribute accesses on an instance of the class (e.g., `obj.attribute`).\n    *   **Behavior:** It intercepts every attribute lookup. To prevent infinite recursion when trying to access attributes within `__getattribute__` itself, you must call the base class's `__getattribute__` (e.g., `object.__getattribute__(self, name)` or `super().__getattribute__(name)`).\n    *   **Relationship with `__getattr__`:** If `__getattribute__` is defined and explicitly calls `__getattr__` or raises an `AttributeError`, then `__getattr__` might be called. Otherwise, `__getattribute__` bypasses `__getattr__`.\n\n*   `__setattr__(self, name, value)`:\n    *   **Purpose:** Called *unconditionally* for all attribute assignments (e.g., `obj.attribute = value`).\n    *   **Behavior:** Allows custom logic for setting attributes. Like `__getattribute__`, you must call `super().__setattr__(name, value)` to avoid infinite recursion when storing the attribute in the instance's dictionary.\n\n*   `__delattr__(self, name)`:\n    *   **Purpose:** Called *unconditionally* for all attribute deletions (e.g., `del obj.attribute`).\n    *   **Behavior:** Allows custom logic for deleting attributes. Must call `super().__delattr__(name)` to avoid infinite recursion.\n\n**Example: Custom Attribute Access with `__getattribute__` and `__getattr__`**\n```python\nclass DynamicAttributes:\n    def __init__(self, data):\n        self._data = data\n\n    def __getattribute__(self, name):\n        # Intercepts all attribute access\n        print(f\"__getattribute__ called for: {name}\")\n        try:\n            # Try to get it normally from the instance or class\n            return super().__getattribute__(name)\n        except AttributeError:\n            # If not found, let __getattr__ handle it as a fallback\n            print(f\"Attribute '{name}' not found via normal lookup.\")\n            raise # Re-raise if __getattr__ also doesn't handle it\n\n    def __getattr__(self, name):\n        # Only called if __getattribute__ raises AttributeError or if attribute is not found otherwise\n        print(f\"__getattr__ called for: {name} (fallback)\")\n        if name in self._data:\n            return f\"Value from data: {self._data[name]}\"\n        raise AttributeError(f\"'{self.__class__.__name__}' object has no attribute '{name}'\")\n\n    def existing_method(self):\n        return \"This is an existing method.\"\n\nd = DynamicAttributes({\"city\": \"London\", \"country\": \"UK\"})\n\nprint(d.existing_method()) # __getattribute__ called, then method is returned.\n# Output: __getattribute__ called for: existing_method\n# Output: This is an existing method.\n\nprint(d.city) # __getattribute__ called, then __getattr__ as fallback.\n# Output: __getattribute__ called for: city\n# Output: Attribute 'city' not found via normal lookup.\n# Output: __getattr__ called for: city (fallback)\n# Output: Value from data: London\n\ntry:\n    print(d.zip_code) # __getattribute__ called, then __getattr__ as fallback, then AttributeError.\nexcept AttributeError as e:\n    print(e)\n# Output: __getattribute__ called for: zip_code\n# Output: Attribute 'zip_code' not found via normal lookup.\n# Output: __getattr__ called for: zip_code (fallback)\n# Output: 'DynamicAttributes' object has no attribute 'zip_code'\n```\nUnderstanding descriptors and custom attribute access provides powerful tools for building flexible, robust, and Pythonic class designs.",
  "external_resources": {
    "free": [
      {
        "title": "Python Descriptors: An Introduction - Real Python",
        "url": "https://realpython.com/python-descriptors/",
        "rating": "4.8/5",
        "price": "0",
        "value_for_money": "High"
      },
      {
        "title": "Descriptor HowTo Guide - Python Official Docs",
        "url": "https://docs.python.org/3/howto/descriptor.html",
        "rating": "5/5",
        "price": "0",
        "value_for_money": "High"
      },
      {
        "title": "Descriptor in Python - GeeksforGeeks",
        "url": "https://www.geeksforgeeks.org/descriptor-in-python/",
        "rating": "4.5/5",
        "price": "0",
        "value_for_money": "Medium"
      },
      {
        "title": "Implementing Descriptors in Your Python Code - HackerNoon",
        "url": "https://hackernoon.com/implementing-descriptors-in-your-python-code",
        "rating": "4/5",
        "price": "0",
        "value_for_money": "Medium"
      },
      {
        "title": "Understanding Descriptors in Python - Medium (Data Overload)",
        "url": "https://medium.com/@data_overload/understanding-descriptors-in-python-6202167d58a5",
        "rating": "4/5",
        "price": "0",
        "value_for_money": "Medium"
      }
    ],
    "paid": [
      {
        "title": "100 Days of Code: The Complete Python Pro Bootcamp for 2025 - Udemy",
        "url": "https://www.udemy.com/course/100-days-of-code/",
        "rating": "4.7/5",
        "price": "$12.99 - $200 (frequently on sale)",
        "value_for_money": "High"
      },
      {
        "title": "Python for Everybody Specialization - Coursera (University of Michigan)",
        "url": "https://www.coursera.org/specializations/python",
        "rating": "4.8/5",
        "price": "$49/month (financial aid available)",
        "value_for_money": "High"
      },
      {
        "title": "Python Descriptors (Video Course) - Real Python",
        "url": "https://realpython.com/courses/python-descriptors/",
        "rating": "4.8/5",
        "price": "Membership Required (~$19.99/month)",
        "value_for_money": "High"
      },
      {
        "title": "Complete Python Bootcamp From Zero to Hero in Python - Udemy",
        "url": "https://www.udemy.com/course/complete-python-bootcamp/",
        "rating": "4.6/5",
        "price": "$12.99 - $199.99 (frequently on sale)",
        "value_for_money": "Medium"
      },
      {
        "title": "Intermediate Python - DataCamp",
        "url": "https://www.datacamp.com/courses/intermediate-python",
        "rating": "4.5/5",
        "price": "Subscription Required (~$25/month)",
        "value_for_money": "Medium"
      }
    ]
  }
}
```

It will store the generated content in the memory cache for easy access and reduce LLM token usage.

```bash
    Please select the concept to learn. Eg. If you want to select day 5 please enter 5:     -1 
    Program done
```

## Working example

