// The router (React Router 7, data mode): the Shell frames every page. Workflow names and run ids are query
// parameters (see routes.ts).
import { useState } from "react";
import { createBrowserRouter, RouterProvider, type RouteObject } from "react-router";
import { Shell } from "./components/Shell";
import { NotFound, RouteError } from "./pages/NotFound";
import { PitWall } from "./pages/PitWall";
import { RunDetail } from "./pages/RunDetail";
import { SettingsPage } from "./pages/Settings";
import { Workflow } from "./pages/Workflow";
import { PATHS } from "./routes";

export const routes: RouteObject[] = [
  {
    path: PATHS.pitWall,
    element: <Shell />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <PitWall /> },
      { path: PATHS.workflow, element: <Workflow /> },
      { path: PATHS.run, element: <RunDetail /> },
      { path: PATHS.settings, element: <SettingsPage /> },
      { path: "*", element: <NotFound /> },
    ],
  },
];

export function App() {
  const [router] = useState(() => createBrowserRouter(routes));
  return <RouterProvider router={router} />;
}
