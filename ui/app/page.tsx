"use client";

import { GridGenerator, Hex, Hexagon, HexGrid, Layout } from "react-hexgrid";
import { useState } from "react";

export default function Home() {
  const board_size = 11;
  // Swap q and r so that q is row (vertical), r is column (horizontal)
  const hexagons = GridGenerator.parallelogram(
    0,
    board_size - 1,
    0,
    board_size - 1
  );

  const [clicked, setClicked] = useState<Record<string, 1 | 2>>({});
  const [currentPlayer, setCurrentPlayer] = useState<1 | 2>(1);

  // Track clicked hexes as a set of string keys "q,r"

  const handleHexClick = (hex: Hex) => {
    const key = `${hex.q},${hex.r}`;
    setClicked((prev) => {

      if (prev[key]) return prev;
      return { ...prev, [key]: currentPlayer };
    });
    setCurrentPlayer((p) => (p === 1 ? 2 : 1));
  };

  return (
    <div className="min-h-screen min-w-screen flex flex-col items-center justify-center bg-stone-800">
      <h1>Hex</h1>
      <HexGrid width={800} height={800} viewBox="-100 -100 200 200">
        <Layout
          size={{ x: 5, y: 5 }}
          spacing={1.06}
            origin={{ x: -50, y: -30 }}
          flat={false}
        >
          {hexagons.map((hex: Hex, index: number) => {
            const key = `${hex.q},${hex.r}`;
            const player = clicked[key];
            let fill;
            if (player === 1) fill = "red";
            else if (player === 2) fill = "blue";
            else fill = "gray";
            return (
              <Hexagon
                key={index}
                q={hex.q}
                r={hex.r}
                s={hex.s}
                textAnchor="middle"
                style={fill ? { fill } : undefined}
                onClick={() => handleHexClick(hex)}
              />
            );
          })}
        </Layout>
      </HexGrid>
    </div>
  );
}
