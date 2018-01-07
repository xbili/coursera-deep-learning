# Neural Network Implementations

These are all implemented from scratch, without much reference to the Jupyter
Notebook provided in the assignments.

## Why?

> What I cannot create, I do not understand.

The above is one of my favorite quotes by Richard Feynman - I learn best through
practice, and I feel that some of the programming assignments in the course was
too easy and spoon-fed to us. That being said, it is understandable that the
assignments (and the course in general) are catered for a wider range of
audience, hence it has to be toned down a little.

Therefore I took the effort to reimplement some of the neural networks in the
course to reinforce my understanding. Also adding a little bit more Python flair
to it.

The concepts from earlier courses are all implemented in `numpy`. Concepts
learnt in course 4 onwards are mostly implemented in `tensorflow`.

## Tests

Models implemented here are all unit tested with the aid of `nose` test package.
Tests are complete with *gradient checking* - a useful concept that was
introduced in the second course.

The tests currently use randomly generated data. Work is in progress for data
that makes more sense - like the ones that the course uses.

To run the unit tests:

```
nosetests --verbose --with-coverage
```

A coverage report will also be generated. What I learnt from looking at the
coverage report is that many times deep learning applications do not benefit
much from code coverage reports since there aren't many branches of execution.

But still it is used as an indication of how much code is tested for these
implementations.

## Contributing

Feel free to provide comments on my implementations and better ways to do it.
I tried to steer away from the 'helper function' method and add some structure
to the program. This might not be the best way and I am always open for
discussion. :)

Also feel free to fork these for your own learning purposes too!

**NOTE**: You won't find direct answers for the Coursera programming assignments
here. Although you can identify some pieces of nuggets here and there. Still
recommended to go through the assignments before looking at these. :)
