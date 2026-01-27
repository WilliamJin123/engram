
- If we see 10 instances of a dog, we should generalize this idea of a "dog" with shared attributes, saving memory by leveraging inheritance
- Crucially, inheritance must save memory / compute / allow us to more easily fetch attributes
- If we see 10 different animals, we could all group them together as well, etc.

- Exceptions => A dog might be special and not have a tail for some reason even though dogs in general have tails (overriding inherited attributes)
- A penguin is a bird but cannot fly

- Nodes are balanced by certainty and surprise. For example, low certainty nodes (workflows, facts, preferences, whatever) are easier to change, but high certainty nodes are harder to change. For example, if we are crucially certain that the Earth is round, it should be very hard to change the node to the Earth is flat, but not impossible based on the amount of evidence. However, if we are not sure what color an object is for example, someone telling us or showing us a color of the object should quickly lock in the color.

- Conditionals ==> I can go outside if the weather is nice. The weather is nice if it is sunny. all encapsulated in nodes with agnostic edges

- Meta-relationships ==> analogies, reasoning about relationships, certainty of relationships, etc.
- Provenance ==> where did i learn this? what context did I learn this? Not always though (humans don't remember hwere they learned 1 + 1 = 2 exactly)
- History ==> How have my understandings changed over time? Why did they change? This would be very important for skills such as coding (we change our approach because things don't work, this is important to remember)


