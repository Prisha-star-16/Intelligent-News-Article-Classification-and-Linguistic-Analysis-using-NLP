import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "NewsLens — News Classification & Linguistic Analysis",
  description: "A 4-class news classifier (AG News) running entirely in your browser, with the NLP analysis behind it.",
};
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (<html lang="en"><body>{children}</body></html>);
}
