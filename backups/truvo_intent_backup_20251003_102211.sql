--
-- PostgreSQL database dump
--

\restrict xWgeoRXHa0yFYL2M5c9GkB41YMmQeIBvOT7MYexNcIf7yaCdznFcep5nCYgXpIo

-- Dumped from database version 17.6 (Debian 17.6-0+deb13u1)
-- Dumped by pg_dump version 17.6 (Debian 17.6-0+deb13u1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: intent_categories; Type: TABLE; Schema: public; Owner: truvo
--

CREATE TABLE public.intent_categories (
    id integer NOT NULL,
    name text NOT NULL,
    description text NOT NULL
);


ALTER TABLE public.intent_categories OWNER TO truvo;

--
-- Name: training_data; Type: TABLE; Schema: public; Owner: truvo
--

CREATE TABLE public.training_data (
    id integer NOT NULL,
    text text NOT NULL,
    intent integer NOT NULL,
    intent_name text NOT NULL,
    confidence real DEFAULT 1.0,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.training_data OWNER TO truvo;

--
-- Name: training_data_id_seq; Type: SEQUENCE; Schema: public; Owner: truvo
--

CREATE SEQUENCE public.training_data_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.training_data_id_seq OWNER TO truvo;

--
-- Name: training_data_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: truvo
--

ALTER SEQUENCE public.training_data_id_seq OWNED BY public.training_data.id;


--
-- Name: training_data id; Type: DEFAULT; Schema: public; Owner: truvo
--

ALTER TABLE ONLY public.training_data ALTER COLUMN id SET DEFAULT nextval('public.training_data_id_seq'::regclass);


--
-- Data for Name: intent_categories; Type: TABLE DATA; Schema: public; Owner: truvo
--

COPY public.intent_categories (id, name, description) FROM stdin;
0	automation	Desktop automation tasks like opening apps, clicking buttons
1	conversation	General chat, greetings, philosophical questions
2	system_info	Local system information queries
3	web_search	Real-time information that needs web search
4	knowledge	General knowledge that can be answered without web search
\.


--
-- Data for Name: training_data; Type: TABLE DATA; Schema: public; Owner: truvo
--

COPY public.training_data (id, text, intent, intent_name, confidence, created_at, updated_at) FROM stdin;
1	open chrome	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
2	open browser	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
3	launch edge	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
4	start firefox	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
5	open notepad	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
6	launch calculator	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
7	start spotify	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
8	open file explorer	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
9	launch terminal	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
10	open command prompt	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
11	click the button	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
12	press enter	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
13	type this text	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
14	scroll down	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
15	scroll up	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
16	close window	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
17	minimize app	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
18	maximize window	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
19	switch tabs	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
20	take screenshot	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
21	save file	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
22	copy text	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
23	paste clipboard	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
24	select all	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
25	drag and drop	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
26	right click	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
27	double click	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
28	open task manager	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
29	can you open chrome	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
30	please launch notepad	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
31	help me open calculator	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
32	start the browser for me	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
33	could you click that button	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
34	can you type this	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
35	please scroll down	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
36	help me close this	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
37	open settings	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
38	launch control panel	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
39	start task manager	0	automation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
40	hello	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
41	hi	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
42	hey	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
43	good morning	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
44	good evening	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
45	how are you	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
46	what's up	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
47	how's it going	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
48	nice to meet you	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
49	what is your name	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
50	tell me about yourself	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
51	how do you work	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
52	what can you do	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
53	are you intelligent	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
54	do you have feelings	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
55	tell me a joke	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
56	make me laugh	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
57	you're funny	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
58	thanks for helping	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
59	you're awesome	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
60	good job	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
61	well done	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
62	i appreciate it	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
63	thank you	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
64	goodbye	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
65	see you later	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
66	have a good day	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
67	take care	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
68	like why me always bro	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
69	bro what's wrong	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
70	dude help me out	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
71	man this is crazy	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
72	yo what's good	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
73	sup friend	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
74	hey buddy	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
75	wassup mate	1	conversation	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
76	show disk space	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
77	check memory usage	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
78	how much ram	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
79	show cpu usage	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
80	system information	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
81	get hardware info	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
82	show battery status	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
83	check uptime	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
84	system uptime	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
85	get windows version	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
86	show network info	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
87	what's my ip address	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
88	show running processes	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
89	list installed programs	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
90	show available drives	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
91	check disk health	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
92	get bios info	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
93	show motherboard details	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
94	check temperature	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
95	show graphics card	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
96	get processor info	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
97	list files in downloads	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
98	show desktop files	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
99	check folder size	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
100	get user accounts	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
101	show startup programs	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
102	check windows updates	2	system_info	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
103	trending news today	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
104	latest news	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
105	breaking news	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
106	current news	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
107	what's happening today	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
108	recent news updates	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
109	trending topics	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
110	viral news	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
111	latest headlines	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
112	weather today	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
113	current weather	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
114	weather forecast	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
115	temperature now	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
116	weather in london	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
117	rain forecast	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
118	stock price	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
119	tesla stock	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
120	bitcoin price	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
121	market news	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
122	crypto prices	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
123	exchange rate	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
124	live sports scores	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
125	cricket score	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
126	football results	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
127	match results	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
128	who won the game	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
129	man of the match	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
130	player of the series	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
131	tournament results	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
132	championship winner	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
133	asia cup final	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
134	trending songs	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
135	viral videos	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
136	youtube trending	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
137	new movie releases	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
138	box office results	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
139	celebrity news	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
140	latest music	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
141	trending hashtags	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
142	social media trends	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
143	current events	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
144	recent updates	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
145	live updates	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
146	real time data	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
147	recent earthquake	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
148	traffic updates	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
149	flight status	3	web_search	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
150	what is python	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
151	explain machine learning	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
152	how does ai work	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
153	what is programming	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
154	define artificial intelligence	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
155	what is the capital of france	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
156	how tall is mount everest	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
157	who invented the computer	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
158	what is quantum physics	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
159	explain photosynthesis	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
160	what is dna	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
161	how do computers work	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
162	what is the internet	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
163	explain gravity	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
164	what is relativity	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
165	how does gps work	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
166	what is blockchain	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
167	explain algorithms	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
168	what is database	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
169	how does wifi work	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
170	what is http	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
171	explain tcp ip	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
172	what is operating system	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
173	how does cpu work	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
174	what is ram	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
175	explain hard drive	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
176	what is software	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
177	define hardware	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
178	what is cloud computing	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
179	explain virtual reality	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
180	what is augmented reality	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
181	how does camera work	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
182	what is semiconductor	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
183	explain neural networks	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
184	what is deep learning	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
185	how does internet work	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
186	what is sql	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
187	explain apis	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
188	what is json	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
189	define xml	4	knowledge	1	2025-10-03 04:30:00.05934	2025-10-03 04:30:00.05934
\.


--
-- Name: training_data_id_seq; Type: SEQUENCE SET; Schema: public; Owner: truvo
--

SELECT pg_catalog.setval('public.training_data_id_seq', 189, true);


--
-- Name: intent_categories intent_categories_pkey; Type: CONSTRAINT; Schema: public; Owner: truvo
--

ALTER TABLE ONLY public.intent_categories
    ADD CONSTRAINT intent_categories_pkey PRIMARY KEY (id);


--
-- Name: training_data training_data_pkey; Type: CONSTRAINT; Schema: public; Owner: truvo
--

ALTER TABLE ONLY public.training_data
    ADD CONSTRAINT training_data_pkey PRIMARY KEY (id);


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: pg_database_owner
--

GRANT ALL ON SCHEMA public TO truvo;


--
-- PostgreSQL database dump complete
--

\unrestrict xWgeoRXHa0yFYL2M5c9GkB41YMmQeIBvOT7MYexNcIf7yaCdznFcep5nCYgXpIo

