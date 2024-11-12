# Association Rule Mining

**WIP**

Association rule mining can be posed as unsupervised learning, where we extract a set of rules from a database. The goal is to find frequently occurring itemsets. It evaluates transactions for correlation and

## Intro

*Association rule learning is a rule-based machine learning method for discovering interesting relations between variables in large data

*For example, the rule {onions,potatoes} => {burger} found in the sales data of a supermarket would indicate that if a customer buys onions and potatoes together, they are likely to also buy hamburger meat. Such information can be used as the basis for decisions about marketing activities such as, e.g., promotional pricing or product placements.

The table above reveals the users that make the market transactions that contain the products purchased. The aim is to put some potential rules from the dataset. Websites like Netflix, IMDB, and Youtube use some more complex type of Apriori model and make some recommendations like “People watch this movie also watch these ones”.

2. How Apriori Works
1.Support: Support refers to the default popularity of an item and can be calculated by finding number of transactions containing a particular item divided by total number of transactions.The number of an Specific Item`s transactions among all of the transactions:

2.Confidence: Confidence refers to the likelihood that an item B is also bought if item A is bought. It can be calculated by finding the number of transactions where A and B are bought together, divided by total number of transactions where A is bought

Example: In the picture above, there are 3 transactions which contain both apple and bear among all apple transactions. Therefore;

            Confidence= 3/4= %75(The possibility of buying bear among apple buyers is %75)-(In apple-bear example in the figure above)
3.Lift: Lift(A -> B) refers to the increase in the ratio of sale of B when A is sold. Lift(A –> B) can be calculated by dividing Confidence(A -> B) divided by Support(B).It is the division of Confidence by Support:

Lift basically tells us that the likelihood of buying a bear and apple together is 1.5 times more than the likelihood of just buying the apple. A Lift of 1 means there is no association between products A and B. Lift of greater than 1 means products A and B are more likely to be bought together. Finally, Lift of less than 1 refers to the case where two products are unlikely to be bought together.