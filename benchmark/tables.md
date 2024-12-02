## ***OpenR*** Benchmark Results

Here we present some preliminary results in the following tables.

<table>
  <thead>
    <tr>
      <th>Generator</th>
      <th>Reward Model</th>
      <th>Search</th>
      <th>Budget</th>
      <th>MATH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="15" style="text-align: center; vertical-align: middle;">
        <img src="https://avatars.githubusercontent.com/u/141221163?s=200&v=4" alt="Qwen2.5 Logo" style="width: 50px; height: 50px;" align="center"><br>
        Qwen2.5-Math-1.5B-Instruct
      </td>
      <td rowspan="5">Math-Sphered-Mistral-7B-PRM</td>
      <td>Greedy</td>
      <td>$2^0$</td>
      <td></td>
    </tr>
    <tr>
      <td>Majority Vote</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Best-of-N</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Beam Search</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>MCTS</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="5">Math-psa-7B</td>
      <td>Greedy</td>
      <td>$2^0$</td>
      <td></td>
    </tr>
    <tr>
      <td>Majority Vote</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Best-of-N</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Beam Search</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>MCTS</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="5">Skywork-o1-PRM-7B</td>
      <td>Greedy</td>
      <td>$2^0$</td>
      <td></td>
    </tr>
    <tr>
      <td>Majority Vote</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Best-of-N</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Beam Search</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>MCTS</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <td></td>
    <tr>
      <td rowspan="15" style="text-align: center; vertical-align: middle;">
        <img src="https://avatars.githubusercontent.com/u/141221163?s=200&v=4" alt="Qwen2.5 Logo" style="width: 50px; height: 50px;" align="center"><br>
        Qwen2.5-Math-7B-Instruct
      </td>
      <td rowspan="5">Math-Sphered-Mistral-7B-PRM</td>
      <td>Greedy</td>
      <td>$2^0$</td>
      <td></td>
    </tr>
    <tr>
      <td>Majority Vote</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Best-of-N</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Beam Search</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>MCTS</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="5">Math-psa-7B</td>
      <td>Greedy</td>
      <td>$2^0$</td>
      <td></td>
    </tr>
    <tr>
      <td>Majority Vote</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Best-of-N</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Beam Search</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>MCTS</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="5">Skywork-o1-PRM-7B</td>
      <td>Greedy</td>
      <td>$2^0$</td>
      <td></td>
    </tr>
    <tr>
      <td>Majority Vote</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Best-of-N</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Beam Search</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>MCTS</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
<td></td>
    <tr>
      <td rowspan="15" style="text-align: center; vertical-align: middle;">
        <img src="https://cdn-avatars.huggingface.co/v1/production/uploads/652260e0728c0b6dc72dd957/lPyWi7-XFLhH3s1c2vnvD.png" alt="Skywork Logo" style="width: 50px; height: 50px;" align="center"><br>
        Skywork-o1-Open-Llama-3.1-8B
      </td>
      <td rowspan="5">Math-Sphered-Mistral-7B-PRM</td>
      <td>Greedy</td>
      <td>$2^0$</td>
      <td></td>
    </tr>
    <tr>
      <td>Majority Vote</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Best-of-N</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Beam Search</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>MCTS</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="5">Math-psa-7B</td>
      <td>Greedy</td>
      <td>$2^0$</td>
      <td></td>
    </tr>
    <tr>
      <td>Majority Vote</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Best-of-N</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Beam Search</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>MCTS</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="5">Skywork-o1-PRM-7B</td>
      <td>Greedy</td>
      <td>$2^0$</td>
      <td></td>
    </tr>
    <tr>
      <td>Majority Vote</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Best-of-N</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>Beam Search</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
    <tr>
      <td>MCTS</td>
      <td>$2^6$</td>
      <td></td>
    </tr>
  </tbody>
</table>

