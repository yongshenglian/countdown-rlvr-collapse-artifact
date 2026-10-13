# Systematically selected collapse-screen outputs

Selection uses recipe, seed, endpoint status, and event time only; response content is never a selection criterion. Within locally archived seeds 3--6, each recipe contributes the lowest-numbered endpoint-collapsed seed at its first sustained event and the lowest-numbered endpoint-healthy seed at step 100 when one exists. A recovered healthy seed is shown both at its earlier event and at step 100.

## Example 1: no DS (rung 2), seed 5, step 30

- Selection role: `first_event_lowest_collapsed_seed`
- Endpoint status: `collapsed`
- Ground truth: `{'numbers': [99, 26, 26], 'target': 100}`
- Training scorer value: `0.1`
- Response characters: `132`
- Source: `ablation/logs/rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed5-s100-20261008-194830.log`

````text
We need to use the numbers [99, 26, 26] exactly once to create an equation that equals 100.
</think>
<answer>(99 - 26) * 26</answer>
````

## Example 2: no DS (rung 2), seed 3, step 100

- Selection role: `lowest_healthy_seed_at_endpoint`
- Endpoint status: `healthy`
- Ground truth: `{'numbers': [99, 26, 26], 'target': 100}`
- Training scorer value: `0.0`
- Response characters: `12252`
- Source: `ablation/logs/rung2-clip0.28-noDS-gen32-train32-n8-resp4096-seed3-s100-20261008-065352.log`

````text
</think>
<think>
We need to create an equation using the numbers [99, 26, 26] that equals 100, with each number used only once.
</think>
<think>
Let's try different combinations of the numbers to see if we can form the equation.
</think>
<think>
One possible approach is to try different combinations of the numbers and see if we can form the equation.
</think>
<think>
Let's start with the numbers 99, 26, and 26.
</think>
<think>
We can try different operations to see if we can get the result of 100.
</think>
<think>
Let's try adding 26 and 26 to 99.
</think>
<think>
99 + 26 + 26 = 151, which is not equal to 100.
</think>
<think>
Let's try subtracting 26 from 99 and then adding 26.
</think>
<think>
99 - 26 + 26 = 99, which is not equal to 100.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26.
</think>
<think>
99 + 26 - 26 = 99, which is not equal to 100.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but in a different order.
</think>
<think>
99 + 26 - 26 = 99, which is not equal to 100.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 100, which is equal to 100.
</think>
<think>
We have found a valid equation that equals 100.
</think>
<think>
The equation is: (99 + 26) - 26
</think>
<think>
Let's double-check the calculation.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26.
</think>
<think>
99 + 26 - 26 = 100, which is equal to 100.
</think>
<think>
We have found a valid equation that equals 100.
</think>
<think>
The equation is: 99 + 26 - 26
</think>
<think>
Let's double-check the calculation.
</think>
<think>
99 + 26 - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but in a different order.
</think>
<think>
99 + 26 - 26 = 100, which is equal to 100.
</think>
<think>
We have found a valid equation that equals 100.
</think>
<think>
The equation is: 99 + 26 - 26
</think>
<think>
Let's double-check the calculation.
</think>
<think>
99 + 26 - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of operations.
</think>
<think>
Let's try adding 99 and 26, and then subtracting 26, but using parentheses to group the operations.
</think>
<think>
(99 + 26) - 26 = 125 - 26 = 99, which is not equal to 100.
</think>
<think>
It seems there was a mistake in the previous calculation. Let's try a different approach.
</think>
<think>
Let's try a different combination of
````

## Example 3: DS, seed 3, step 20

- Selection role: `first_event_lowest_collapsed_seed`
- Endpoint status: `collapsed`
- Ground truth: `{'numbers': [99, 26, 26], 'target': 100}`
- Training scorer value: `0.0`
- Response characters: `33`
- Source: `ablation/logs/rung3-DS-clip0.28-gen192-train32-n8-resp4096-seed3-s100-20261008-120120.log`

````text
<answer> (99 - 26) + 26 </answer>
````

## Example 4: A1, seed 3, step 25

- Selection role: `first_event_lowest_collapsed_seed`
- Endpoint status: `collapsed`
- Ground truth: `{'numbers': [99, 26, 26], 'target': 100}`
- Training scorer value: `0.0`
- Response characters: `33`
- Source: `ablation/logs/rungA1-accOnlyReward-maskTruncated-clip0.28-noDS-gen32-train32-n8-resp4096-seed3-s100-20261008-100803.log`

````text
<answer> (99 - 26) + 26 </answer>
````

## Example 5: A1 + KL 0.01, seed 3, step 30

- Selection role: `first_event_lowest_collapsed_seed`
- Endpoint status: `collapsed`
- Ground truth: `{'numbers': [99, 26, 26], 'target': 100}`
- Training scorer value: `0.0`
- Response characters: `31`
- Source: `ablation/logs/rungA1KL-accOnlyReward-maskTruncated-kl0.01-clip0.28-noDS-gen32-train32-n8-resp4096-seed3-s100-20261008-105838.log`

````text
<answer> 99 - 26 + 26 </answer>
````

## Example 6: A1 + KL 0.01, seed 4, step 25

- Selection role: `recovered_seed_at_first_event`
- Endpoint status: `healthy`
- Ground truth: `{'numbers': [99, 26, 26], 'target': 100}`
- Training scorer value: `0.0`
- Response characters: `31`
- Source: `ablation/logs/rungA1KL-accOnlyReward-maskTruncated-kl0.01-clip0.28-noDS-gen32-train32-n8-resp4096-seed4-s100-20261008-172451.log`

````text
<answer> 99 - 26 + 26 </answer>
````

## Example 7: A1 + KL 0.01, seed 4, step 100

- Selection role: `same_recovered_seed_at_endpoint`
- Endpoint status: `healthy`
- Ground truth: `{'numbers': [99, 26, 26], 'target': 100}`
- Training scorer value: `0.0`
- Response characters: `173`
- Source: `ablation/logs/rungA1KL-accOnlyReward-maskTruncated-kl0.01-clip0.28-noDS-gen32-train32-n8-resp4096-seed4-s100-20261008-172451.log`

````text
<answer> (99 - 26) + 26 </answer>
- First, we subtract 26 from 99 to get 73.
- Then, we add 26 to 73 to get 100.
- Therefore, the equation that equals 100 is (99 - 26) + 26.
````
