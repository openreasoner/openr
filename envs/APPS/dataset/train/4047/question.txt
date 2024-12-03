```if-not:rust
Your task is to write a function `toLeetSpeak` that converts a regular english sentence to Leetspeak.
```
```if:rust
Your task is to write a function `to_leet_speak` that converts a regular english sentence to Leetspeak.
```

More about LeetSpeak You can read at wiki -> https://en.wikipedia.org/wiki/Leet

Consider only uppercase letters (no lowercase letters, no numbers) and spaces.

For example:

```if-not:rust
~~~
toLeetSpeak("LEET") returns "1337"
~~~
```
```if:rust
~~~
to_leet_speak("LEET") returns "1337"
~~~
```

In this kata we use a simple LeetSpeak dialect. Use this alphabet:

```
{
  A : '@',
  B : '8',
  C : '(',
  D : 'D',
  E : '3',
  F : 'F',
  G : '6',
  H : '#',
  I : '!',
  J : 'J',
  K : 'K',
  L : '1',
  M : 'M',
  N : 'N',
  O : '0',
  P : 'P',
  Q : 'Q',
  R : 'R',
  S : '$',
  T : '7',
  U : 'U',
  V : 'V',
  W : 'W',
  X : 'X',
  Y : 'Y',
  Z : '2'
}
```