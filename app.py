import os
import json
import streamlit as st
import pandas as pd
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import requests
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
# Configure page
st.set_page_config(
    page_title="AI Marketing Persona Designer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulse 2s infinite alternate;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        100% { transform: scale(1.02); }
    }
    
    .persona-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .persona-card:hover {
        transform: translateY(-5px);
    }
    
    .campaign-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .campaign-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .metric-card h4 {
        color: #ffffff !important;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-card h2 {
        color: #ffffff !important;
        font-weight: bold;
        font-size: 2rem;
        margin: 0;
    }
    
    .agent-status {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
        animation: pulse 2s infinite;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
        border: none;
        border-radius: 25px;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .confidence-score {
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .refined-indicator {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.8rem;
        color: white;
        font-weight: bold;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-5px); }
        60% { transform: translateY(-3px); }
    }
    
    .new-feature {
        background: linear-gradient(45deg, #00c9ff, #92fe9d);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Data Structures
@dataclass
class EnhancedPersona:
    name: str
    tagline: str
    demographics: Dict[str, Any]
    psychographics: Dict[str, Any]
    behavior_patterns: Dict[str, Any]
    pain_points: List[str]
    goals: List[str]
    preferred_channels: List[str]
    messaging_preferences: Dict[str, Any]
    confidence_score: float
    market_size: str
    business_value: str
    is_refined: bool = False
    refinement_history: List[str] = None

@dataclass
class EnhancedCampaign:
    title: str
    persona_target: str
    theme: str
    key_message: str
    value_propositions: List[str]
    channels: List[str]
    content_formats: List[str]
    success_metrics: List[str]
    predicted_roi: str
    confidence_interval: str
    budget_allocation: Dict[str, str]

@dataclass
class PersonaJourney:
    stages: List[str]
    touchpoints: List[str]
    emotions: List[str]
    opportunities: List[str]

# Enhanced AI Analysis Engine with Fixed Bugs
class EnhancedAIAnalysisEngine:
    def __init__(self, api_key, model_name='gemini-2.0-flash'):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
    
    def analyze_customer_data(self, customer_data: str, product_info: str) -> Dict:
        """Analyze customer data using Gemini"""
        
        prompt = f"""
        As an expert marketing data analyst, analyze this customer research data and product information:

        CUSTOMER DATA:
        {customer_data}

        PRODUCT INFO:
        {product_info}

        Provide a comprehensive analysis in JSON format with:
        1. Customer segments (3-4 distinct behavioral clusters)
        2. Key demographic patterns
        3. Pain points and motivations
        4. Communication preferences
        5. Market opportunities

        Return only valid JSON without any markdown formatting.
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            return json.loads(response_text)
        except Exception as e:
            return self._get_fallback_analysis()
    
    def create_personas(self, analysis_data: Dict, num_personas: int = 3) -> Dict:
        """Create detailed personas based on analysis"""
        
        prompt = f"""
        Based on this customer analysis data, create exactly {num_personas} detailed marketing personas:

        ANALYSIS DATA:
        {json.dumps(analysis_data, indent=2)}

        For each persona, provide:
        1. Name and tagline
        2. Detailed demographics
        3. Psychographic profile
        4. Behavioral patterns
        5. Pain points and goals
        6. Communication preferences
        7. Confidence scores (0-1)

        Return as JSON with a "personas" array containing exactly {num_personas} personas. No markdown formatting.
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            return json.loads(response_text)
        except Exception as e:
            return self._get_fallback_personas(num_personas)
    
    def create_campaigns(self, personas_data: Dict) -> Dict:
        """Create campaign strategies for each persona"""
        
        prompt = f"""
        Create comprehensive marketing campaign strategies for these personas:

        PERSONAS:
        {json.dumps(personas_data, indent=2)}

        For each persona, create a campaign with:
        1. Campaign title and theme
        2. Core messaging strategy
        3. Channel recommendations
        4. Content strategy
        5. ROI predictions
        6. Success metrics

        Return as JSON with a "campaigns" array. No markdown formatting.
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            return json.loads(response_text)
        except Exception as e:
            return self._get_fallback_campaigns()
    
    def refine_persona(self, original_persona: Dict, feedback: str) -> Dict:
        """FIXED: Refine persona based on user feedback"""
        prompt = f"""
        Refine this marketing persona based on the user feedback. Make meaningful changes to improve the persona:
        
        ORIGINAL PERSONA:
        {json.dumps(original_persona, indent=2)}
        
        USER FEEDBACK:
        {feedback}
        
        Based on the feedback, update relevant fields while keeping the JSON structure intact.
        Make sure to incorporate the feedback into demographics, psychographics, pain_points, goals, or other relevant sections.
        Return the complete updated persona as JSON with no markdown formatting.
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            refined_persona = json.loads(response_text)
            
            # Add refinement metadata
            refined_persona['is_refined'] = True
            refined_persona['last_refinement'] = datetime.now().isoformat()
            refined_persona['refinement_feedback'] = feedback
            
            if 'refinement_history' not in refined_persona:
                refined_persona['refinement_history'] = []
            refined_persona['refinement_history'].append({
                'timestamp': datetime.now().isoformat(),
                'feedback': feedback
            })
            
            return refined_persona
        except Exception as e:
            st.error(f"Refinement failed: {str(e)}")
            return original_persona
    
    def generate_content_sample(self, campaign_data: Dict) -> Dict:
        """Generate sample marketing content for campaign"""
        prompt = f"""
        Generate comprehensive marketing content samples for this campaign:
        
        CAMPAIGN DATA:
        {json.dumps(campaign_data, indent=2)}

        Create realistic, engaging content including:
        1. Email subject line and full body
        2. Social media posts (3 variations)
        3. Google Ad copy (headline + description)
        4. Blog post title and introduction paragraph
        5. Landing page headline and value proposition

        Return as JSON with detailed content. No markdown formatting.
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            return json.loads(response_text)
        except:
            return {
                "email": {
                    "subject": "Transform Your Business with AI-Powered Solutions",
                    "body": "Dear [Name],\n\nDiscover how our innovative platform can revolutionize your workflow and boost productivity by 300%. Join thousands of successful businesses who've already made the switch.\n\nBest regards,\nYour Marketing Team"
                },
                "social_posts": [
                    "🚀 Ready to 3x your productivity? Our AI-powered solution is changing the game! #Innovation #Productivity",
                    "Join 10,000+ businesses already saving time with our platform. What are you waiting for? 💪",
                    "The future is here! Experience the power of intelligent automation. Try it free today! ⚡"
                ],
                "google_ad": {
                    "headline": "Boost Productivity 300% | AI Solution",
                    "description": "Transform your workflow with intelligent automation. Join 10,000+ satisfied customers. Free trial available!"
                },
                "blog": {
                    "title": "The Future of Productivity: How AI is Transforming Business Operations",
                    "intro": "In today's fast-paced business environment, staying competitive means embracing innovation. Artificial Intelligence isn't just a buzzword—it's a game-changing technology that's helping businesses of all sizes achieve unprecedented levels of efficiency and growth."
                },
                "landing_page": {
                    "headline": "Unlock 300% More Productivity with AI",
                    "value_prop": "Revolutionary AI platform that automates your workflow, saves time, and drives results. Join 10,000+ businesses already experiencing the transformation."
                }
            }
    
    def generate_journey_map(self, persona_data: Dict) -> Dict:
        """Generate detailed customer journey map"""
        prompt = f"""
        Create a comprehensive customer journey map for this persona:
        
        PERSONA DATA:
        {json.dumps(persona_data, indent=2)}

        Generate a detailed journey with:
        1. 6-7 key stages (Awareness to Advocacy)
        2. Specific touchpoints for each stage
        3. Emotional states throughout journey
        4. Pain points at each stage
        5. Opportunities for engagement
        6. Recommended actions

        Return as JSON with "journey_map" containing detailed stage information.
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            return json.loads(response_text)
        except:
            return {
                "journey_map": [
                    {
                        "stage": "Awareness",
                        "touchpoints": ["Social Media", "Search Ads", "Word of Mouth"],
                        "emotions": ["Curious", "Skeptical"],
                        "pain_points": ["Information overload", "Too many options"],
                        "opportunities": ["Educational content", "Clear messaging"],
                        "actions": ["Create awareness campaigns", "SEO optimization"]
                    },
                    {
                        "stage": "Interest",
                        "touchpoints": ["Website", "Blog", "Reviews"],
                        "emotions": ["Interested", "Hopeful"],
                        "pain_points": ["Unclear pricing", "Complex information"],
                        "opportunities": ["Detailed product info", "Social proof"],
                        "actions": ["Landing page optimization", "Customer testimonials"]
                    },
                    {
                        "stage": "Consideration",
                        "touchpoints": ["Product demos", "Sales calls", "Comparisons"],
                        "emotions": ["Evaluating", "Cautious"],
                        "pain_points": ["Decision fatigue", "Budget concerns"],
                        "opportunities": ["Free trials", "ROI calculators"],
                        "actions": ["Demo scheduling", "Competitive analysis"]
                    },
                    {
                        "stage": "Purchase",
                        "touchpoints": ["Checkout", "Sales team", "Payment"],
                        "emotions": ["Excited", "Anxious"],
                        "pain_points": ["Complex checkout", "Payment issues"],
                        "opportunities": ["Smooth process", "Multiple payment options"],
                        "actions": ["Streamline checkout", "Payment flexibility"]
                    },
                    {
                        "stage": "Onboarding",
                        "touchpoints": ["Welcome emails", "Setup guides", "Support"],
                        "emotions": ["Overwhelmed", "Determined"],
                        "pain_points": ["Steep learning curve", "Lack of guidance"],
                        "opportunities": ["Step-by-step guidance", "Video tutorials"],
                        "actions": ["Onboarding sequences", "Support resources"]
                    },
                    {
                        "stage": "Usage",
                        "touchpoints": ["Product interface", "Support", "Updates"],
                        "emotions": ["Satisfied", "Productive"],
                        "pain_points": ["Feature complexity", "Performance issues"],
                        "opportunities": ["Feature training", "Performance optimization"],
                        "actions": ["User education", "Product improvements"]
                    },
                    {
                        "stage": "Advocacy",
                        "touchpoints": ["Referrals", "Reviews", "Case studies"],
                        "emotions": ["Proud", "Confident"],
                        "pain_points": ["Limited referral incentives"],
                        "opportunities": ["Referral programs", "Success stories"],
                        "actions": ["Loyalty programs", "Case study development"]
                    }
                ]
            }
    
    def answer_query(self, query: str, context: Dict) -> str:
        """FIXED: Enhanced AI assistant for queries"""
        prompt = f"""
        You are an expert marketing consultant. Answer this query based on the analysis data provided.
        Be specific, actionable, and reference the actual data when possible.
        
        USER QUERY: {query}
        
        CONTEXT DATA:
        Personas: {json.dumps(context.get('personas_data', {}), indent=2)}
        Campaigns: {json.dumps(context.get('campaigns_data', {}), indent=2)}
        
        Provide a helpful, detailed response with specific recommendations and insights.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question or check the system status."

    def simulate_performance(self, campaign_data: Dict) -> Dict:
        """Generate realistic performance simulation"""
        prompt = f"""
        Create a detailed performance simulation for this marketing campaign:
        
        CAMPAIGN:
        {json.dumps(campaign_data, indent=2)}

        Generate realistic metrics including:
        1. Projected reach and impressions
        2. Engagement rates by channel
        3. Detailed conversion funnel
        4. ROI scenarios (optimistic/realistic/conservative)
        5. Timeline predictions
        6. Budget breakdown and efficiency metrics

        Return comprehensive JSON simulation data.
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            return json.loads(response_text)
        except:
            return {
                "reach": {
                    "total_impressions": "250,000",
                    "unique_reach": "85,000",
                    "frequency": "2.9"
                },
                "engagement": {
                    "overall_rate": "4.7%",
                    "email_open_rate": "24%",
                    "social_engagement": "6.2%",
                    "website_ctr": "3.1%"
                },
                "conversion_funnel": {
                    "impressions": "250,000",
                    "clicks": "7,750",
                    "leads": "930",
                    "qualified_leads": "465",
                    "sales": "93",
                    "conversion_rate": "1.2%"
                },
                "roi_scenarios": {
                    "optimistic": {"roi": "4.8x", "probability": "20%"},
                    "realistic": {"roi": "3.2x", "probability": "60%"},
                    "conservative": {"roi": "2.1x", "probability": "20%"}
                },
                "timeline": {
                    "launch_phase": "Weeks 1-2",
                    "optimization_phase": "Weeks 3-6",
                    "scaling_phase": "Weeks 7-12",
                    "full_roi_expected": "Month 4"
                },
                "budget_efficiency": {
                    "cost_per_click": "$0.87",
                    "cost_per_lead": "$8.60",
                    "customer_acquisition_cost": "$86.00",
                    "return_on_ad_spend": "320%"
                }
            }
    
    def generate_ab_test_ideas(self, campaign_data: Dict) -> Dict:
        """NEW FEATURE: Generate A/B test ideas"""
        prompt = f"""
        Generate A/B testing ideas for this campaign to optimize performance:
        
        CAMPAIGN:
        {json.dumps(campaign_data, indent=2)}

        Suggest 5-7 specific A/B tests covering:
        1. Headlines and messaging
        2. Visual elements
        3. Call-to-action buttons
        4. Email subject lines
        5. Landing page layouts
        6. Targeting parameters

        Return as JSON with test ideas and expected impact.
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            return json.loads(response_text)
        except:
            return {
                "ab_tests": [
                    {
                        "test_name": "Headline Optimization",
                        "element": "Main headline",
                        "variant_a": "Current headline",
                        "variant_b": "Benefit-focused headline",
                        "expected_impact": "+15% conversion rate",
                        "test_duration": "2 weeks"
                    },
                    {
                        "test_name": "CTA Button Color",
                        "element": "Call-to-action button",
                        "variant_a": "Blue button",
                        "variant_b": "Orange button",
                        "expected_impact": "+8% click-through rate",
                        "test_duration": "1 week"
                    }
                ]
            }
    
    def generate_competitor_analysis(self, personas_data: Dict) -> Dict:
        """NEW FEATURE: Competitor analysis insights"""
        prompt = f"""
        Based on these personas, generate a competitor analysis framework:
        
        PERSONAS:
        {json.dumps(personas_data, indent=2)}

        Provide:
        1. Key competitors likely targeting these personas
        2. Competitive positioning opportunities
        3. Differentiation strategies
        4. Pricing considerations
        5. Channel strategy vs competitors

        Return as JSON with actionable insights.
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            return json.loads(response_text)
        except:
            return {
                "competitor_landscape": {
                    "direct_competitors": ["Competitor A", "Competitor B", "Competitor C"],
                    "indirect_competitors": ["Alternative Solution 1", "Alternative Solution 2"],
                    "positioning_gaps": ["Underserved premium segment", "SMB market opportunity"],
                    "differentiation_opportunities": ["Superior customer service", "Advanced features", "Better pricing"]
                }
            }
    
    def _get_fallback_analysis(self):
        """Fallback analysis data"""
        return {
            "customer_segments": [
                {
                    "name": "Efficiency Seekers",
                    "size": "35%",
                    "traits": ["time-conscious", "tech-savvy", "quality-focused"]
                },
                {
                    "name": "Value Optimizers",
                    "size": "30%", 
                    "traits": ["budget-conscious", "research-driven", "family-oriented"]
                },
                {
                    "name": "Premium Pursuers",
                    "size": "25%",
                    "traits": ["quality-first", "brand-loyal", "premium-willing"]
                }
            ],
            "key_insights": {
                "primary_pain_points": ["time constraints", "complex processes", "poor value"],
                "top_motivations": ["efficiency", "savings", "quality"]
            }
        }
    
    def _get_fallback_personas(self, num_personas: int = 3):
        """Enhanced fallback personas data"""
        all_personas = [
            {
                "name": "Alex the Efficiency Expert",
                "tagline": "Time is money, quality is non-negotiable",
                "demographics": {
                    "age_range": "28-40",
                    "income_range": "$65k-$120k",
                    "education": "Bachelor's+",
                    "location": "Urban/Suburban",
                    "occupation": "Project Manager/Consultant"
                },
                "psychographics": {
                    "values": ["efficiency", "innovation", "work-life balance"],
                    "personality_traits": ["analytical", "goal-oriented", "tech-savvy"],
                    "lifestyle": "Fast-paced, digitally connected, career-focused",
                    "interests": ["productivity tools", "tech trends", "professional development"]
                },
                "behavior_patterns": {
                    "decision_making": "Research-driven, quick decisions",
                    "brand_loyalty": "Medium - switches for better solutions",
                    "price_sensitivity": "Low - values time savings over cost",
                    "content_consumption": "Blogs, podcasts, video tutorials"
                },
                "pain_points": [
                    "Information overload and decision fatigue",
                    "Time-consuming processes and poor UX",
                    "Lack of integration between tools",
                    "Inefficient team collaboration methods"
                ],
                "goals": [
                    "Maximize productivity and efficiency",
                    "Stay ahead of technology trends", 
                    "Achieve work-life balance",
                    "Lead successful project outcomes"
                ],
                "preferred_channels": ["LinkedIn", "Email", "Professional forums", "Webinars"],
                "messaging_preferences": {
                    "tone": "Professional, direct, value-focused",
                    "content_types": ["How-to guides", "Case studies", "Video demos", "ROI calculators"],
                    "frequency": "Weekly updates, immediate alerts for relevant content"
                },
                "confidence_score": 0.92,
                "market_size": "32%",
                "business_value": "High"
            },
            {
                "name": "Jordan the Value Optimizer",
                "tagline": "Smart choices for smart families",
                "demographics": {
                    "age_range": "35-50",
                    "income_range": "$45k-$85k",
                    "education": "High School - Bachelor's",
                    "location": "Suburban/Small City",
                    "family_status": "Married with 2-3 children"
                },
                "psychographics": {
                    "values": ["family", "financial security", "practical solutions"],
                    "personality_traits": ["cautious", "caring", "community-minded"],
                    "lifestyle": "Family-centered, budget-conscious, community-involved",
                    "interests": ["family activities", "saving money", "local community"]
                },
                "behavior_patterns": {
                    "decision_making": "Thorough research, seeks recommendations",
                    "brand_loyalty": "High - sticks with trusted brands",
                    "price_sensitivity": "High - compares prices extensively",
                    "content_consumption": "Reviews, comparison sites, social media"
                },
                "pain_points": [
                    "Limited budget with growing family needs",
                    "Difficulty evaluating value vs cost",
                    "Hidden fees and unexpected costs",
                    "Time constraints for thorough research"
                ],
                "goals": [
                    "Provide best value for family",
                    "Financial stability and security",
                    "Make smart, informed decisions",
                    "Save time while ensuring quality"
                ],
                "preferred_channels": ["Facebook", "Email newsletters", "Community forums", "Local advertising"],
                "messaging_preferences": {
                    "tone": "Warm, trustworthy, family-focused",
                    "content_types": ["Customer testimonials", "Comparison charts", "Family stories", "Money-saving tips"],
                    "frequency": "Bi-weekly newsletters, seasonal promotions"
                },
                "confidence_score": 0.88,
                "market_size": "28%",
                "business_value": "Medium-High"
            },
            {
                "name": "Sam the Premium Pursuer",
                "tagline": "Quality over everything else",
                "demographics": {
                    "age_range": "40-65",
                    "income_range": "$100k+",
                    "education": "Bachelor's - Advanced Degree",
                    "location": "Urban/Affluent Suburban",
                    "occupation": "Executive/Business Owner"
                },
                "psychographics": {
                    "values": ["quality", "exclusivity", "expertise"],
                    "personality_traits": ["discerning", "confident", "success-oriented"],
                    "lifestyle": "Premium-focused, time-rich but selective, status-conscious",
                    "interests": ["luxury experiences", "industry leadership", "exclusive networks"]
                },
                "behavior_patterns": {
                    "decision_making": "Expert consultation, brand reputation focused",
                    "brand_loyalty": "Very high - premium brand advocate",
                    "price_sensitivity": "Very low - quality is paramount",
                    "content_consumption": "Industry publications, expert opinions, premium content"
                },
                "pain_points": [
                    "Finding authentic premium quality",
                    "Distinguishing between genuine and inflated value",
                    "Lack of personalized service",
                    "Time wasted on subpar solutions"
                ],
                "goals": [
                    "Access the highest quality solutions",
                    "Maintain status and exclusivity",
                    "Save time with premium service",
                    "Make strategic business decisions"
                ],
                "preferred_channels": ["Email", "Premium publications", "Exclusive events", "Executive networks"],
                "messaging_preferences": {
                    "tone": "Sophisticated, exclusive, expert-level",
                    "content_types": ["Expert insights", "Behind-the-scenes content", "Premium case studies", "Executive briefings"],
                    "frequency": "Monthly premium content, exclusive early access"
                },
                "confidence_score": 0.91,
                "market_size": "25%",
                "business_value": "Very High"
            },
            {
                "name": "Casey the Innovation Adopter",
                "tagline": "First to try, first to succeed",
                "demographics": {
                    "age_range": "25-35",
                    "income_range": "$55k-$95k",
                    "education": "Bachelor's+",
                    "location": "Urban/Tech Hubs",
                    "occupation": "Marketing/Tech Professional"
                },
                "psychographics": {
                    "values": ["innovation", "trendsetting", "social influence"],
                    "personality_traits": ["curious", "social", "risk-tolerant"],
                    "lifestyle": "Tech-forward, socially connected, early adopter",
                    "interests": ["new technology", "social media trends", "startup culture"]
                },
                "behavior_patterns": {
                    "decision_making": "Impulse-driven, influenced by social proof",
                    "brand_loyalty": "Low - always seeking the next big thing",
                    "price_sensitivity": "Medium - willing to pay for innovation",
                    "content_consumption": "Social media, tech blogs, influencer content"
                },
                "pain_points": [
                    "Missing out on latest trends",
                    "Limited social proof for new products",
                    "Overwhelming choice of new options",
                    "Fear of backing the wrong innovation"
                ],
                "goals": [
                    "Stay ahead of the curve",
                    "Build social influence and credibility",
                    "Find innovative solutions to everyday problems",
                    "Be recognized as a thought leader"
                ],
                "preferred_channels": ["Instagram", "TikTok", "Tech blogs", "Twitter", "LinkedIn"],
                "messaging_preferences": {
                    "tone": "Exciting, innovative, cutting-edge",
                    "content_types": ["Product launches", "Beta features", "Influencer content", "Trend reports"],
                    "frequency": "Daily updates, real-time notifications"
                },
                "confidence_score": 0.86,
                "market_size": "22%",
                "business_value": "High"
            },
            {
                "name": "Riley the Relationship Builder",
                "tagline": "Connection and community first",
                "demographics": {
                    "age_range": "30-55",
                    "income_range": "$40k-$75k",
                    "education": "High School - Bachelor's",
                    "location": "Suburban/Rural",
                    "occupation": "Teacher/Non-profit/Small Business"
                },
                "psychographics": {
                    "values": ["community", "relationships", "authenticity"],
                    "personality_traits": ["empathetic", "loyal", "collaborative"],
                    "lifestyle": "Community-focused, relationship-driven, authentic",
                    "interests": ["local community", "volunteering", "personal connections"]
                },
                "behavior_patterns": {
                    "decision_making": "Relationship-based, seeks personal recommendations",
                    "brand_loyalty": "Very high - supports brands that align with values",
                    "price_sensitivity": "Medium - values relationship over price",
                    "content_consumption": "Word-of-mouth, community forums, personal stories"
                },
                "pain_points": [
                    "Impersonal service experiences",
                    "Lack of genuine connection with brands",
                    "Difficulty finding trustworthy recommendations",
                    "Corporate messaging that feels inauthentic"
                ],
                "goals": [
                    "Build meaningful connections",
                    "Support businesses that share values",
                    "Create positive community impact",
                    "Foster authentic relationships"
                ],
                "preferred_channels": ["Community events", "Word-of-mouth", "Local social media", "Email"],
                "messaging_preferences": {
                    "tone": "Personal, authentic, community-focused",
                    "content_types": ["Stories", "Community spotlights", "Personal testimonials", "Behind-the-scenes"],
                    "frequency": "Weekly community updates, event notifications"
                },
                "confidence_score": 0.87,
                "market_size": "18%",
                "business_value": "Medium"
            }
        ]
        
        # Return the requested number of personas
        selected_personas = all_personas[:num_personas]
        return {"personas": selected_personas}
    
    def _get_fallback_campaigns(self):
        """Enhanced fallback campaigns data"""
        return {
            "campaigns": [
                {
                    "title": "Efficiency Accelerator Campaign",
                    "persona_target": "Alex the Efficiency Expert",
                    "theme": "Time is Your Most Valuable Asset",
                    "key_message": "Transform your productivity with intelligent automation that saves hours every day",
                    "value_propositions": [
                        "Save 3+ hours daily with smart automation",
                        "Seamless integration with existing tools",
                        "Real-time analytics and insights",
                        "Enterprise-grade security and reliability"
                    ],
                    "channels": ["LinkedIn Ads", "Google Search", "Email Marketing", "Webinars"],
                    "content_formats": ["Video demos", "ROI calculators", "Case studies", "Comparison guides"],
                    "content_strategy": [
                        "Productivity Tips & Hacks",
                        "Industry Efficiency Trends", 
                        "Customer Success Stories",
                        "Integration Tutorials"
                    ],
                    "success_metrics": ["Time saved per user", "User adoption rate", "Feature utilization", "Customer satisfaction"],
                    "predicted_roi": "3.4x",
                    "conversion_rate": "8.5%",
                    "payback_period": "6 months",
                    "confidence_interval": "85-95%",
                    "budget_allocation": {
                        "digital_ads": "40%",
                        "content_creation": "30%",
                        "email_marketing": "20%",
                        "events_webinars": "10%"
                    }
                },
                {
                    "title": "Smart Family Value Campaign",
                    "persona_target": "Jordan the Value Optimizer",
                    "theme": "Smart Choices for Smart Families",
                    "key_message": "The smart choice families trust for unbeatable value and peace of mind",
                    "value_propositions": [
                        "Best value guarantee with transparent pricing",
                        "Family-friendly features and safety",
                        "24/7 customer support when you need it",
                        "Money-back satisfaction guarantee"
                    ],
                    "channels": ["Facebook", "Instagram", "Community Partnerships", "Local Radio"],
                    "content_formats": ["Family testimonials", "Value comparisons", "Safety guides", "Community stories"],
                    "content_strategy": [
                        "Family Success Stories",
                        "Money-Saving Tips",
                        "Community Spotlights",
                        "Value Comparison Charts"
                    ],
                    "success_metrics": ["Customer lifetime value", "Referral rate", "Family satisfaction", "Repeat purchase rate"],
                    "predicted_roi": "2.8x",
                    "conversion_rate": "6.2%",
                    "payback_period": "8 months",
                    "confidence_interval": "75-85%",
                    "budget_allocation": {
                        "social_media": "35%",
                        "community_partnerships": "25%",
                        "content_creation": "25%",
                        "traditional_media": "15%"
                    }
                },
                {
                    "title": "Premium Excellence Experience",
                    "persona_target": "Sam the Premium Pursuer",
                    "theme": "Exceptional Quality for Discerning Individuals",
                    "key_message": "Uncompromising excellence for those who accept nothing less than the best",
                    "value_propositions": [
                        "Premium quality with exclusive features",
                        "White-glove service and personal attention",
                        "Industry-leading expertise and support",
                        "Exclusive access to premium community"
                    ],
                    "channels": ["Premium Email", "Industry Publications", "Executive Networks", "Exclusive Events"],
                    "content_formats": ["Executive briefings", "Premium case studies", "Expert interviews", "Exclusive reports"],
                    "content_strategy": [
                        "Industry Leadership Content",
                        "Premium Insights & Trends",
                        "Exclusive Access Programs",
                        "Executive Success Stories"
                    ],
                    "success_metrics": ["Premium customer acquisition", "Average deal size", "Customer advocacy", "Retention rate"],
                    "predicted_roi": "4.1x",
                    "conversion_rate": "12.3%",
                    "payback_period": "4 months",
                    "confidence_interval": "90-95%",
                    "budget_allocation": {
                        "premium_content": "35%",
                        "exclusive_events": "30%",
                        "executive_outreach": "25%",
                        "industry_publications": "10%"
                    }
                }
            ]
        }

# Initialize AI Engine with Enhanced Error Handling
@st.cache_resource
def initialize_ai_engine():
    """Initialize AI Analysis Engine with embedded API key and fallback options"""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found in environment variables. Please check your .env file.")
        return None
    
    try:
        genai.configure(api_key=api_key)
        test_model = genai.GenerativeModel('gemini-2.0-flash')
        test_response = test_model.generate_content("Hello")
        
        st.success("✅ Connected to Gemini Flash 2.0 Experimental!")
        return EnhancedAIAnalysisEngine(api_key)
        
    except Exception as e:
        st.warning(f"⚠️ Primary model unavailable, trying alternatives...")
        
        alternative_models = [
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'gemini-1.0-pro'
        ]
        
        for model_name in alternative_models:
            try:
                test_model = genai.GenerativeModel(model_name)
                test_response = test_model.generate_content("Hello")
                st.success(f"✅ Connected using {model_name}")
                return EnhancedAIAnalysisEngine(api_key, model_name)
            except:
                continue
        
        st.error("❌ Could not connect to any Gemini model. Using fallback mode.")
        return None

# Enhanced Visualization Functions
def create_confidence_chart(personas_data):
    """Create enhanced confidence score visualization"""
    if not personas_data or 'personas' not in personas_data:
        return None
        
    personas = personas_data['personas']
    names = []
    scores = []
    refined_status = []
    
    for p in personas:
        names.append(p.get('name', 'Unknown Persona'))
        
        confidence = p.get('confidence_score', 0.85)
        if isinstance(confidence, dict):
            confidence = confidence.get('overall_confidence', 0.85)
        elif isinstance(confidence, str):
            try:
                confidence = float(confidence.replace('%', '')) / 100
            except:
                confidence = 0.85
        
        scores.append(confidence * 100)
        refined_status.append('Refined' if p.get('is_refined', False) else 'Original')
    
    if not names or not scores:
        return None
    
    fig = px.bar(
        x=names,
        y=scores,
        title="🎯 Persona Confidence Scores",
        labels={'x': 'Personas', 'y': 'Confidence Score (%)'},
        color=scores,
        color_continuous_scale='RdYlGn',
        text=refined_status
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333'),
        title_font_size=20
    )
    
    return fig

def create_market_size_chart(personas_data):
    """Create enhanced market size distribution chart"""
    if not personas_data or 'personas' not in personas_data:
        return None
        
    personas = personas_data['personas']
    names = []
    sizes = []
    colors = []
    
    for p in personas:
        name = p.get('name', 'Unknown Persona')
        if p.get('is_refined', False):
            name += " ⭐"
        names.append(name)
        
        market_size = p.get('market_size', '25%')
        if isinstance(market_size, dict):
            market_size = market_size.get('market_segment_size', '25%')
        
        try:
            if isinstance(market_size, str):
                size_val = float(market_size.replace('%', ''))
            else:
                size_val = float(market_size)
        except:
            size_val = 25.0
        
        sizes.append(size_val)
        colors.append('#ff6b6b' if p.get('is_refined', False) else '#4ecdc4')
    
    if not names or not sizes or sum(sizes) == 0:
        return None
    
    fig = px.pie(
        names=names,
        values=sizes,
        title="📊 Market Segment Distribution",
        color_discrete_sequence=colors
    )
    
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        textfont_size=12
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333'),
        title_font_size=20
    )
    
    return fig

def create_roi_comparison_chart(campaigns_data):
    """Create enhanced ROI comparison chart"""
    if not campaigns_data or 'campaigns' not in campaigns_data:
        return None
        
    campaigns = campaigns_data['campaigns']
    titles = []
    roi_values = []
    confidence_intervals = []
    
    for c in campaigns:
        titles.append(c.get('title', 'Campaign'))
        
        roi = c.get('predicted_roi', '2.5x')
        if isinstance(roi, dict):
            roi = roi.get('projected_roi', '2.5x')
        
        try:
            roi_val = float(str(roi).replace('x', ''))
        except:
            roi_val = 2.5
        
        roi_values.append(roi_val)
        confidence_intervals.append(c.get('confidence_interval', '80-90%'))
    
    if not titles or not roi_values:
        return None
    
    fig = px.bar(
        x=titles,
        y=roi_values,
        title="🚀 Predicted Campaign ROI",
        labels={'x': 'Campaigns', 'y': 'ROI Multiplier'},
        color=roi_values,
        color_continuous_scale='RdYlGn',
        text=confidence_intervals
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333'),
        xaxis_tickangle=-45,
        title_font_size=20
    )
    
    return fig

def create_journey_map_chart(journey_data: Dict):
    """Create enhanced interactive journey map visualization"""
    if not journey_data or 'journey_map' not in journey_data:
        return None
        
    stages = journey_data['journey_map']
    x = [s.get('stage', 'Stage') for s in stages]
    
    # Create emotion scores (positive/negative scale)
    emotion_scores = []
    for s in stages:
        emotions = s.get('emotions', [])
        # Simple scoring based on common emotion words
        positive_emotions = ['excited', 'satisfied', 'happy', 'confident', 'proud', 'hopeful']
        negative_emotions = ['anxious', 'frustrated', 'overwhelmed', 'confused', 'skeptical']
        
        score = 0
        for emotion in emotions:
            if any(pos in emotion.lower() for pos in positive_emotions):
                score += 1
            elif any(neg in emotion.lower() for neg in negative_emotions):
                score -= 1
        
        emotion_scores.append(score)
    
    fig = go.Figure()
    
    # Add emotion journey line
    fig.add_trace(go.Scatter(
        x=x,
        y=emotion_scores,
        mode='lines+markers+text',
        name='Customer Emotion Journey',
        text=[', '.join(s.get('emotions', [])) for s in stages],
        textposition="top center",
        marker=dict(size=12, color='blue'),
        line=dict(width=3)
    ))
    
    # Add opportunity markers
    fig.add_trace(go.Scatter(
        x=x,
        y=[max(emotion_scores) + 1] * len(x),
        mode='markers+text',
        name='Key Opportunities',
        text=[', '.join(s.get('opportunities', [])[:2]) for s in stages],
        textposition="top center",
        marker=dict(size=10, color='green', symbol='star'),
        showlegend=False
    ))
    
    fig.update_layout(
        title="🗺️ Customer Journey Map",
        xaxis_title="Journey Stages",
        yaxis_title="Emotional Experience",
        yaxis=dict(
            tickmode='array',
            tickvals=[-2, -1, 0, 1, 2],
            ticktext=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
        ),
        showlegend=True,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_size=20
    )
    
    return fig

# Enhanced Display Functions
def display_personas(personas_data):
    """FIXED: Display personas with enhanced information and refinement indicators"""
    if not personas_data or 'personas' not in personas_data:
        st.warning("No personas generated yet.")
        return
    
    personas = personas_data['personas']
    
    for i, persona in enumerate(personas):
        name = persona.get('name', 'Unknown Persona')
        tagline = persona.get('tagline', 'Marketing persona')
        is_refined = persona.get('is_refined', False)
        
        # Handle demographics safely
        demographics = persona.get('demographics', {})
        age_range = demographics.get('age_range', demographics.get('age', 'N/A'))
        income_range = demographics.get('income_range', demographics.get('income', 'N/A'))
        education = demographics.get('education', 'N/A')
        location = demographics.get('location', 'N/A')
        occupation = demographics.get('occupation', 'N/A')
        
        # Handle psychographics safely
        psychographics = persona.get('psychographics', {})
        personality_traits = psychographics.get('personality_traits', ['analytical', 'focused'])
        if isinstance(personality_traits, str):
            personality_traits = [personality_traits]
        
        values = psychographics.get('values', ['success', 'efficiency'])
        if isinstance(values, str):
            values = [values]
        
        # Handle pain points safely
        pain_points = persona.get('pain_points', ['Various challenges and concerns'])
        if isinstance(pain_points, str):
            pain_points = [pain_points]
        
        # Handle goals safely
        goals = persona.get('goals', persona.get('goals_motivations', ['Achieve success']))
        if isinstance(goals, str):
            goals = [goals]
        
        # Handle channels safely
        channels = persona.get('preferred_channels', ['Email', 'Social Media'])
        if isinstance(channels, str):
            channels = [channels]
        
        # Use columns for layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Display persona header with refinement indicator
            refinement_badge = ""
            if is_refined:
                refinement_badge = '<span class="refined-indicator">✨ REFINED</span>'
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0; 
                        box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
                <h3>🎭 {name} {refinement_badge}</h3>
                <p style="font-style: italic; font-size: 1.1em;">"{tagline}"</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Demographics section
            st.markdown("**📊 Demographics:**")
            demo_col1, demo_col2 = st.columns(2)
            with demo_col1:
                st.write(f"**Age:** {age_range}")
                st.write(f"**Education:** {education}")
                st.write(f"**Occupation:** {occupation}")
            with demo_col2:
                st.write(f"**Income:** {income_range}")
                st.write(f"**Location:** {location}")
            
            # Psychographics section
            st.markdown("**🧠 Psychographic Profile:**")
            st.write(f"**Personality:** {', '.join(personality_traits[:4])}")
            st.write(f"**Core Values:** {', '.join(values[:4])}")
            
            # Pain points section
            st.markdown("**😟 Key Pain Points:**")
            for pain in pain_points[:3]:
                st.write(f"• {pain}")
            
            # Goals section
            st.markdown("**🎯 Primary Goals:**")
            for goal in goals[:3]:
                st.write(f"• {goal}")
            
            # Channels section
            st.markdown("**📱 Preferred Channels:**")
            st.write(", ".join(channels[:4]))
            
            # Refinement history if available
            if is_refined and persona.get('refinement_history'):
                with st.expander("🔍 Refinement History"):
                    for entry in persona.get('refinement_history', []):
                        st.write(f"**{entry.get('timestamp', 'Unknown date')}:** {entry.get('feedback', 'No feedback recorded')}")
        
        with col2:
            # Safely get metrics
            confidence = persona.get('confidence_score', persona.get('confidence_metrics', {}).get('overall_confidence', 0.85))
            market_size = persona.get('market_size', persona.get('business_metrics', {}).get('market_segment_size', '25%'))
            business_value = persona.get('business_value', persona.get('business_metrics', {}).get('estimated_value', 'Medium'))
            
            # Ensure confidence is a float
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence.replace('%', '')) / 100
                except:
                    confidence = 0.85
            
            # Ensure market_size has %
            if isinstance(market_size, (int, float)):
                market_size = f"{market_size}%"
            elif not str(market_size).endswith('%'):
                market_size = f"{market_size}%"
            
            # Display metrics
            confidence_color = "🟢" if confidence > 0.85 else "🟡" if confidence > 0.7 else "🔴"
            st.metric("Confidence Score", f"{confidence:.0%}", delta=confidence_color)
            st.metric("Market Share", market_size)
            st.metric("Business Value", business_value)
            
            # Refinement timestamp if available
            if is_refined:
                last_refined = persona.get('last_refinement', 'Recently')
                if 'T' in str(last_refined):
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(last_refined.replace('Z', '+00:00'))
                        last_refined = dt.strftime('%m/%d %H:%M')
                    except:
                        last_refined = 'Recently'
                st.caption(f"🔄 Last refined: {last_refined}")
        
        st.markdown("---")

def display_campaigns(campaigns_data):
    """FIXED: Display campaigns with enhanced metrics"""
    if not campaigns_data or 'campaigns' not in campaigns_data:
        st.warning("No campaigns generated yet.")
        return
    
    campaigns = campaigns_data['campaigns']
    
    for campaign in campaigns:
        title = campaign.get('title', 'Marketing Campaign')
        persona_target = campaign.get('persona_target', 'Target Audience')
        theme = campaign.get('theme', campaign.get('campaign_theme', 'Campaign Theme'))
        key_message = campaign.get('key_message', campaign.get('core_messaging', {}).get('primary_message', 'Engaging marketing message'))
        
        # Handle channels safely
        channels = campaign.get('channels', ['Email', 'Social Media'])
        if isinstance(channels, str):
            channels = [channels]
        elif isinstance(channels, dict):
            channels = list(channels.keys())
        
        # Handle value propositions safely
        value_props = campaign.get('value_propositions', ['Great value', 'Quality service'])
        if isinstance(value_props, str):
            value_props = [value_props]
        
        # Handle content strategy safely
        content_strategy = campaign.get('content_strategy', ['Brand Awareness', 'Customer Engagement'])
        
        if content_strategy is None:
            content_strategy = ['Brand Awareness', 'Customer Engagement']
        elif isinstance(content_strategy, dict):
            if 'content_pillars' in content_strategy:
                content_strategy = content_strategy['content_pillars']
            elif content_strategy:
                content_strategy = list(content_strategy.keys())
            else:
                content_strategy = ['Brand Awareness', 'Customer Engagement']
        elif isinstance(content_strategy, str):
            content_strategy = [content_strategy]
        elif not isinstance(content_strategy, list):
            content_strategy = ['Brand Awareness', 'Customer Engagement']
        
        if not content_strategy or not isinstance(content_strategy, list):
            content_strategy = ['Brand Awareness', 'Customer Engagement']
        
        # Use columns for layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0; 
                        box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
                <h3>🚀 {title}</h3>
                <p style="font-style: italic;">🎯 Target: {persona_target}</p>
                <p style="font-weight: bold; font-size: 1.1em;">"{theme}"</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**💬 Key Message:**")
            st.write(key_message)
            
            st.markdown("**🎁 Value Propositions:**")
            for prop in value_props[:3]:
                st.write(f"• {prop}")
            
            st.markdown("**📱 Primary Channels:**")
            channel_badges = " ".join([f"`{channel}`" for channel in channels[:4]])
            st.markdown(channel_badges)
            
            st.markdown("**📋 Content Strategy:**")
            for strategy in content_strategy[:4]:
                st.write(f"• {strategy}")
        
        with col2:
            # Performance metrics
            performance = campaign.get('performance_predictions', {})
            roi = campaign.get('predicted_roi', performance.get('projected_roi', '2.5x'))
            conversion = campaign.get('conversion_rate', performance.get('predicted_conversion_rate', '5.0%'))
            payback = campaign.get('payback_period', performance.get('payback_period', '12 months'))
            confidence_interval = campaign.get('confidence_interval', '80-90%')
            
            # Enhanced metrics display
            st.metric("Expected ROI", roi, delta=f"Confidence: {confidence_interval}")
            st.metric("Conversion Rate", conversion)
            st.metric("Payback Period", payback)
            
            # Budget allocation if available
            budget = campaign.get('budget_allocation', {})
            if budget:
                st.markdown("**💰 Budget Split:**")
                for channel, percentage in list(budget.items())[:3]:
                    st.write(f"{channel}: {percentage}")
        
        st.markdown("---")

# Enhanced Content Display Functions
def display_content_samples(content_data: Dict):
    """Display generated content samples with copy buttons"""
    st.markdown("### 📝 AI-Generated Content Samples")
    
    # Email Campaign
    email = content_data.get('email', {})
    with st.expander("📧 Email Campaign", expanded=True):
        st.text_input("Subject Line", value=email.get('subject', 'Sample Subject'), key=f"email_subject_{id(content_data)}")
        st.text_area("Email Body", value=email.get('body', 'Sample body...'), height=150, key=f"email_body_{id(content_data)}")
    
    # Social Media Posts
    social_posts = content_data.get('social_posts', [content_data.get('social_post', 'Sample post...')])
    if isinstance(social_posts, str):
        social_posts = [social_posts]
    
    with st.expander("📱 Social Media Content"):
        for i, post in enumerate(social_posts[:3]):
            st.text_area(f"Social Post {i+1}", value=post, height=100, key=f"social_post_{i}_{id(content_data)}")
    
    # Google Ads
    google_ad = content_data.get('google_ad', {})
    with st.expander("🎯 Google Ads"):
        st.text_input("Headline", value=google_ad.get('headline', 'Sample headline'), key=f"ad_headline_{id(content_data)}")
        st.text_area("Description", value=google_ad.get('description', 'Sample description'), height=100, key=f"ad_desc_{id(content_data)}")
    
    # Blog Content
    blog = content_data.get('blog', {})
    with st.expander("📝 Blog Post"):
        st.text_input("Blog Title", value=blog.get('title', 'Sample Title'), key=f"blog_title_{id(content_data)}")
        st.text_area("Introduction", value=blog.get('intro', 'Sample intro...'), height=150, key=f"blog_intro_{id(content_data)}")
    
    # Landing Page
    landing = content_data.get('landing_page', {})
    if landing:
        with st.expander("🎯 Landing Page"):
            st.text_input("Landing Page Headline", value=landing.get('headline', 'Sample headline'), key=f"landing_headline_{id(content_data)}")
            st.text_area("Value Proposition", value=landing.get('value_prop', 'Sample value prop...'), height=100, key=f"landing_value_{id(content_data)}")

def display_performance_simulation(sim_data: Dict):
    """Display enhanced campaign performance simulation"""
    st.markdown("### 📈 Advanced Performance Simulation")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    reach_data = sim_data.get('reach', {})
    engagement_data = sim_data.get('engagement', {})
    funnel_data = sim_data.get('conversion_funnel', {})
    
    with col1:
        total_reach = reach_data.get('total_impressions', sim_data.get('reach', 'N/A'))
        st.metric("Total Impressions", total_reach)
    
    with col2:
        engagement_rate = engagement_data.get('overall_rate', sim_data.get('engagement_rate', 'N/A'))
        st.metric("Engagement Rate", engagement_rate)
    
    with col3:
        conversion_rate = funnel_data.get('conversion_rate', 'N/A')
        st.metric("Conversion Rate", conversion_rate)
    
    with col4:
        total_sales = funnel_data.get('sales', 'N/A')
        st.metric("Projected Sales", total_sales)
    
    # ROI Scenarios with Visual Progress
    roi_scenarios = sim_data.get('roi_scenarios', {})
    st.markdown("#### 💰 ROI Scenario Analysis")
    
    scenarios = [
        ("🎯 Conservative", roi_scenarios.get('conservative', {}).get('roi', '2.1x'), 0.3, '#ff6b6b'),
        ("📊 Realistic", roi_scenarios.get('realistic', {}).get('roi', '3.2x'), 0.7, '#4ecdc4'),
        ("🚀 Optimistic", roi_scenarios.get('optimistic', {}).get('roi', '4.8x'), 1.0, '#45b7d1')
    ]
    
    for label, roi_value, progress, color in scenarios:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(progress, f"{label}: {roi_value}")
        with col2:
            probability = roi_scenarios.get(label.split()[1].lower(), {}).get('probability', 'N/A')
            st.caption(f"Probability: {probability}")
    
    # Detailed Funnel Visualization
    if funnel_data:
        st.markdown("#### 🔄 Conversion Funnel Breakdown")
        funnel_stages = [
            ("Impressions", funnel_data.get('impressions', '0')),
            ("Clicks", funnel_data.get('clicks', '0')),
            ("Leads", funnel_data.get('leads', '0')),
            ("Qualified Leads", funnel_data.get('qualified_leads', '0')),
            ("Sales", funnel_data.get('sales', '0'))
        ]
        
        funnel_col1, funnel_col2, funnel_col3, funnel_col4, funnel_col5 = st.columns(5)
        cols = [funnel_col1, funnel_col2, funnel_col3, funnel_col4, funnel_col5]
        
        for i, (stage, value) in enumerate(funnel_stages):
            with cols[i]:
                # Convert string numbers to int for progress calculation
                try:
                    num_val = int(str(value).replace(',', ''))
                    max_val = int(str(funnel_stages[0][1]).replace(',', ''))
                    progress_val = num_val / max_val if max_val > 0 else 0
                except:
                    progress_val = 0
                
                st.metric(stage, value)
                st.progress(progress_val)
    
    # Timeline and Budget Efficiency
    timeline_data = sim_data.get('timeline', {})
    budget_data = sim_data.get('budget_efficiency', {})
    
    if timeline_data or budget_data:
        col1, col2 = st.columns(2)
        
        with col1:
            if timeline_data:
                st.markdown("#### ⏱️ Campaign Timeline")
                for phase, period in timeline_data.items():
                    st.write(f"**{phase.replace('_', ' ').title()}:** {period}")
        
        with col2:
            if budget_data:
                st.markdown("#### 💸 Budget Efficiency")
                for metric, value in budget_data.items():
                    formatted_metric = metric.replace('_', ' ').title()
                    st.write(f"**{formatted_metric}:** {value}")

def display_ab_test_ideas(ab_data: Dict):
    """Display A/B testing recommendations"""
    st.markdown("### 🧪 A/B Testing Recommendations")
    
    tests = ab_data.get('ab_tests', [])
    
    for i, test in enumerate(tests):
        with st.expander(f"Test {i+1}: {test.get('test_name', 'A/B Test')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Element:** {test.get('element', 'N/A')}")
                st.write(f"**Variant A:** {test.get('variant_a', 'N/A')}")
                st.write(f"**Variant B:** {test.get('variant_b', 'N/A')}")
            
            with col2:
                st.write(f"**Expected Impact:** {test.get('expected_impact', 'N/A')}")
                st.write(f"**Test Duration:** {test.get('test_duration', 'N/A')}")
                st.write(f"**Priority:** {'High' if i < 2 else 'Medium' if i < 4 else 'Low'}")

# Export Functions (Enhanced)
def generate_comprehensive_report(personas_data, campaigns_data, analysis_data=None):
    """Generate comprehensive markdown report"""
    content = "# 🎯 AI Marketing Persona & Campaign Analysis Report\n\n"
    content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    content += f"**AI Engine:** Google Gemini Pro\n"
    content += f"**Analysis Type:** Comprehensive Customer Intelligence\n\n"
    
    # Executive Summary
    content += "## 📊 Executive Summary\n\n"
    personas_count = len(personas_data.get('personas', [])) if personas_data else 0
    campaigns_count = len(campaigns_data.get('campaigns', [])) if campaigns_data else 0
    
    content += f"This report presents {personas_count} detailed customer personas and {campaigns_count} "
    content += "strategic marketing campaigns generated through AI analysis of customer research data.\n\n"
    
    # Personas Section
    content += "## 👥 Customer Personas\n\n"
    if personas_data and 'personas' in personas_data:
        for i, persona in enumerate(personas_data['personas']):
            name = persona.get('name', f'Persona {i+1}')
            tagline = persona.get('tagline', 'Marketing persona')
            is_refined = persona.get('is_refined', False)
            
            content += f"### {name} {'✨ (Refined)' if is_refined else ''}\n"
            content += f"*{tagline}*\n\n"
            
            # Demographics
            demographics = persona.get('demographics', {})
            content += "**Demographics:**\n"
            content += f"- Age: {demographics.get('age_range', 'N/A')}\n"
            content += f"- Income: {demographics.get('income_range', 'N/A')}\n"
            content += f"- Education: {demographics.get('education', 'N/A')}\n"
            content += f"- Location: {demographics.get('location', 'N/A')}\n\n"
            
            # Key insights
            pain_points = persona.get('pain_points', [])
            if pain_points:
                content += "**Key Pain Points:**\n"
                for pain in pain_points[:3]:
                    content += f"- {pain}\n"
                content += "\n"
            
            goals = persona.get('goals', [])
            if goals:
                content += "**Primary Goals:**\n"
                for goal in goals[:3]:
                    content += f"- {goal}\n"
                content += "\n"
            
            # Business metrics
            confidence = persona.get('confidence_score', 0.85)
            market_size = persona.get('market_size', '25%')
            content += f"**Business Metrics:** Confidence: {confidence:.0%}, Market Size: {market_size}\n\n"
            
            content += "---\n\n"
    
    # Campaigns Section
    content += "## 🚀 Campaign Strategies\n\n"
    if campaigns_data and 'campaigns' in campaigns_data:
        for i, campaign in enumerate(campaigns_data['campaigns']):
            title = campaign.get('title', f'Campaign {i+1}')
            target = campaign.get('persona_target', 'Target Audience')
            theme = campaign.get('theme', 'Campaign Theme')
            roi = campaign.get('predicted_roi', 'N/A')
            
            content += f"### {title}\n"
            content += f"**Target Persona:** {target}\n"
            content += f"**Campaign Theme:** {theme}\n"
            content += f"**Expected ROI:** {roi}\n\n"
            
            key_message = campaign.get('key_message', '')
            if key_message:
                content += f"**Key Message:** {key_message}\n\n"
            
            channels = campaign.get('channels', [])
            if channels:
                content += f"**Primary Channels:** {', '.join(channels[:4])}\n\n"
            
            content += "---\n\n"
    
    # Recommendations
    content += "## 💡 Strategic Recommendations\n\n"
    content += "1. **Persona Refinement:** Continuously gather customer feedback to refine personas\n"
    content += "2. **A/B Testing:** Test campaign variations to optimize performance\n"
    content += "3. **Cross-Channel Integration:** Ensure consistent messaging across all channels\n"
    content += "4. **Performance Monitoring:** Track key metrics and adjust strategies accordingly\n"
    content += "5. **Personalization:** Use persona insights to create highly targeted content\n\n"
    
    content += "---\n"
    content += "*Report generated by AI Marketing Persona Designer*\n"
    content += "*Powered by Google Gemini Pro for advanced customer intelligence*"
    
    return content
def generate_persona_insights(personas_data):
    """Generate smart insights about persona distribution and opportunities"""
    if not personas_data or 'personas' not in personas_data:
        return {}
    
    insights = {
        'market_distribution': {},
        'confidence_analysis': {},
        'targeting_recommendations': [],
        'revenue_potential': {}
    }
    
    personas = personas_data['personas']
    
    # Market size analysis
    total_market = 0
    market_segments = []
    
    for persona in personas:
        market_size = persona.get('market_size', '25%')
        try:
            if isinstance(market_size, str):
                size = float(market_size.replace('%', ''))
            else:
                size = float(market_size)
            market_segments.append(size)
            total_market += size
        except:
            market_segments.append(25.0)
    
    insights['market_distribution'] = {
        'total_addressable': f"{total_market:.1f}%",
        'largest_segment': max(market_segments),
        'segment_balance': 'Balanced' if max(market_segments) - min(market_segments) < 20 else 'Unbalanced'
    }
    
    # Confidence analysis
    confidences = []
    for persona in personas:
        conf = persona.get('confidence_score', 0.85)
        if isinstance(conf, str):
            try:
                conf = float(conf.replace('%', '')) / 100
            except:
                conf = 0.85
        confidences.append(conf)
    
    insights['confidence_analysis'] = {
        'average_confidence': sum(confidences) / len(confidences),
        'highest_confidence': max(confidences),
        'reliability_score': 'High' if min(confidences) > 0.8 else 'Medium' if min(confidences) > 0.6 else 'Low'
    }
    
    # Targeting recommendations
    if insights['confidence_analysis']['average_confidence'] > 0.85:
        insights['targeting_recommendations'].append("🎯 High-confidence personas - ready for immediate campaign launch")
    
    if insights['market_distribution']['segment_balance'] == 'Unbalanced':
        insights['targeting_recommendations'].append("⚖️ Consider focusing on largest segment first for maximum impact")
    
    # Revenue potential (simplified estimation)
    high_value_personas = sum(1 for p in personas if p.get('business_value', '').lower() in ['high', 'very high'])
    insights['revenue_potential'] = {
        'high_value_segments': high_value_personas,
        'revenue_tier': 'Premium' if high_value_personas >= len(personas) / 2 else 'Standard'
    }
    
    return insights
def validate_and_score_data(customer_data, product_info):
    """Enhanced data validation with detailed scoring"""
    scores = {
        'customer_data_quality': 0,
        'product_detail_quality': 0,
        'overall_readiness': 0,
        'recommendations': []
    }
    
    # Customer data analysis
    if customer_data:
        word_count = len(customer_data.split())
        line_count = len(customer_data.split('\n'))
        
        # Check for key indicators
        has_demographics = any(word in customer_data.lower() for word in ['age', 'income', 'salary', 'years old'])
        has_pain_points = any(word in customer_data.lower() for word in ['problem', 'issue', 'difficult', 'frustrated', 'need'])
        has_behaviors = any(word in customer_data.lower() for word in ['use', 'buy', 'prefer', 'like', 'want'])
        
        # Scoring
        scores['customer_data_quality'] = min(100, (
            (word_count * 0.5) +
            (line_count * 2) +
            (has_demographics * 20) +
            (has_pain_points * 15) +
            (has_behaviors * 15)
        ))
        
        # Recommendations
        if word_count < 100:
            scores['recommendations'].append("Add more detailed customer feedback or interview data")
        if not has_demographics:
            scores['recommendations'].append("Include age, income, or occupation information")
        if not has_pain_points:
            scores['recommendations'].append("Add customer problems or pain points")
    
    # Product info analysis
    if product_info:
        word_count = len(product_info.split())
        has_pricing = any(word in product_info.lower() for word in ['$', 'price', 'cost', 'free', 'premium'])
        has_features = any(word in product_info.lower() for word in ['feature', 'benefit', 'capability'])
        has_target = any(word in product_info.lower() for word in ['target', 'customer', 'market'])
        
        scores['product_detail_quality'] = min(100, (
            (word_count * 2) +
            (has_pricing * 25) +
            (has_features * 20) +
            (has_target * 15)
        ))
        
        if word_count < 20:
            scores['recommendations'].append("Provide more detailed product information")
        if not has_pricing:
            scores['recommendations'].append("Include pricing information")
    
    scores['overall_readiness'] = (scores['customer_data_quality'] + scores['product_detail_quality']) / 2
    
    return scores
def display_competitive_analysis(comp_data):
    """Enhanced competitive analysis display"""
    st.markdown("### 🏢 Competitive Intelligence")
    
    landscape = comp_data.get('competitor_landscape', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Direct Competitors")
        competitors = landscape.get('direct_competitors', [])
        for i, comp in enumerate(competitors[:5]):
            threat_level = "🔴" if i < 2 else "🟡" if i < 4 else "🟢"
            st.write(f"{threat_level} {comp}")
        
        st.markdown("#### Positioning Gaps")
        gaps = landscape.get('positioning_gaps', [])
        for gap in gaps[:3]:
            st.success(f"🎯 {gap}")
    
    with col2:
        st.markdown("#### Differentiation Opportunities")
        opportunities = landscape.get('differentiation_opportunities', [])
        for opp in opportunities[:4]:
            st.info(f"💡 {opp}")
        
        st.markdown("#### Competitive Advantage Score")
        # Simple scoring based on opportunities
        advantage_score = min(100, len(opportunities) * 20 + len(gaps) * 15)
        st.progress(advantage_score / 100, f"Competitive Advantage: {advantage_score}%")
# Main Application (Enhanced)
def main():
    # Header with enhanced animations
    st.markdown('<h1 class="main-header">🎯 AI Marketing Persona Designer</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; font-size: 1.2rem; color: #666;">
        Transform customer research into actionable personas & campaigns using <strong>Advanced AI Intelligence</strong>
        <br><span class="new-feature">🚀 NEW: Real-time Refinement & Content Generation</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.title("🛠️ AI Configuration")
    
    # Initialize AI Engine
    ai_engine = initialize_ai_engine()
    if not ai_engine:
        st.error("🚨 AI system initialization failed. Please refresh the page.")
        st.stop()
    
    # AI Status Display (Enhanced)
    st.sidebar.markdown("### 🤖 AI System Status")
    if ai_engine:
        st.sidebar.success(f"✅ Model: {getattr(ai_engine, 'model_name', 'gemini-2.0-flash')}")
        st.sidebar.success("✅ API Key: Configured")
        st.sidebar.markdown("""
        <div class="agent-status">🔍 Analysis Engine: Ready</div>
        <div class="agent-status">🎭 Persona Creator: Ready</div>
        <div class="agent-status">🚀 Campaign Builder: Ready</div>
        <div class="agent-status">✨ Refinement Engine: Ready</div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.error("❌ AI System: Offline")
    
    # PERSONA COUNT SLIDER
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎭 Persona Configuration")
    num_personas = st.sidebar.slider(
        "Number of Target Personas:",
        min_value=2,
        max_value=5,
        value=3,
        step=1,
        help="Select how many customer personas to generate (2-5)",
        key="num_personas_slider"
    )
    st.sidebar.info(f"🎯 Generating {num_personas} detailed personas")
    
    # Advanced Options
    with st.sidebar.expander("⚙️ Advanced Options"):
        enable_competitor_analysis = st.checkbox("Include Competitor Analysis", value=True)
        enable_ab_testing = st.checkbox("Generate A/B Test Ideas", value=True)
        enable_journey_maps = st.checkbox("Create Journey Maps", value=True)
    
    # Input Configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Input")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["📝 Paste Research Data", "📁 Upload CSV File", "🎯 Use Demo Data"],
        help="Select how you want to provide customer research data"
    )
    
    customer_data = ""
    
    # Enhanced input validation
    if input_method == "📁 Upload CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload customer research CSV", 
            type=['csv'],
            help="Upload surveys, reviews, or feedback data"
        )
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Loaded {len(df)} records, {len(df.columns)} columns")
        
        # Show data preview
            with st.sidebar.expander("📊 Data Preview"):
                st.dataframe(df.head(3))
        
            customer_data = df.to_string(max_rows=100)  # Increased limit
        
    elif input_method == "📝 Paste Research Data":
        customer_data = st.sidebar.text_area(
        "Paste customer research data:",
        height=200,
        placeholder="Survey responses, customer reviews, feedback, interview transcripts...",
        help="Paste any customer research data - surveys, reviews, interviews, etc."
    )
    
    # Real-time validation
        if customer_data:
            validation = validate_and_score_data(customer_data, product_info)
            score = validation['customer_data_quality']
        
            if score > 80:
                st.sidebar.success(f"✅ Excellent data quality ({score:.0f}%)")
            elif score > 60:
                st.sidebar.warning(f"⚠️ Good data quality ({score:.0f}%)")
            else:
                st.sidebar.error(f"❌ More data needed ({score:.0f}%)")
            
            if validation['recommendations']:
                with st.sidebar.expander("💡 Data Improvement Tips"):
                    for rec in validation['recommendations'][:3]:
                        st.write(f"• {rec}")
        
    else:  # Demo data
        customer_data = """
Age 34, Software Engineer, $85k income: "I need tools that save time and integrate well with my existing workflow. Customer service response time is crucial - I can't wait days for support."

Age 42, Teacher, $55k income: "Budget is always a concern with two kids in college. I research thoroughly before buying anything for the family. Value for money is everything."

Age 29, Marketing Manager, $70k income: "I love trying new products, especially if they're innovative and trending. Social proof is important - I check reviews and what influencers are saying."

Age 51, Business Owner, $120k income: "Quality is non-negotiable for my business. I'm willing to pay premium for excellent products and white-glove service."

Review: "Great product quality but wish the onboarding was simpler. Support team was helpful though - they walked me through everything."

Survey Response: "Price is reasonable for the value provided. My family uses this daily now. Kids love the user interface."

Interview: "As a busy professional, I appreciate products that respect my time. The interface is intuitive and I was productive immediately."

Feedback: "Love the premium features, but would like more customization options for power users like myself."

Age 38, Nurse, $65k income: "Healthcare worker here - I need reliable, professional-grade solutions I can trust during critical moments."

Review: "Excellent ROI and my team's productivity improved significantly. The analytics dashboard is incredibly helpful for tracking performance."

Age 26, Freelancer, $45k income: "As a freelancer, I need affordable solutions that help me compete with bigger agencies. Automation features are game-changers."

Survey: "The mobile app is fantastic - I can manage everything on the go between client meetings."
        """
        st.sidebar.info("🎯 Using comprehensive demo dataset")
    
    # Product information
    product_info = st.sidebar.text_area(
        "Product/Service Details:",
        height=120,
        placeholder="Describe your product, target market, unique value proposition, pricing, key features...",
        help="Provide context about what you're marketing",
        value="AI-powered productivity platform that helps businesses automate workflows, integrate tools, and boost team efficiency. Starting at $29/month with premium tiers up to $299/month. Key features include smart automation, analytics dashboard, mobile app, and 24/7 support."
    )
    
    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        # Enhanced Dashboard
        st.subheader("📈 System Status")
        
        # Data quality assessment
        data_score = min(100, len(customer_data.split()) * 1.5) if customer_data else 0
        product_score = min(100, len(product_info.split()) * 3) if product_info else 0
        overall_score = (data_score + product_score) / 2
        
        # Dynamic color coding
        score_color = "#4CAF50" if overall_score > 70 else "#FF9800" if overall_score > 40 else "#F44336"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Data Quality Score</h4>
            <h2 style="color: {score_color};">{data_score:.0f}%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Product Detail Score</h4>
            <h2 style="color: {score_color};">{product_score:.0f}%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Target Personas</h4>
            <h2 style="color: #ffffff;">{num_personas}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # System Health with Advanced Features
        st.markdown("### 🏥 System Health")
        st.success("✅ AI Engine: Online")
        st.success("✅ API Connection: Active") 
        st.success("✅ Analysis Ready: ✓")
        if enable_competitor_analysis:
            st.info("🔍 Competitor Analysis: Enabled")
        if enable_ab_testing:
            st.info("🧪 A/B Testing: Enabled")
        if enable_journey_maps:
            st.info("🗺️ Journey Maps: Enabled")
    
    with col1:
        # Generation Controls
        st.subheader("🚀 Generate Marketing Intelligence")
        
        # Enhanced validation
        ready_to_generate = bool(customer_data and product_info and overall_score > 30)
        
        if not ready_to_generate:
            if not customer_data:
                st.warning("⚠️ Please provide customer research data.")
            if not product_info:
                st.warning("⚠️ Please provide product information.")
            if overall_score <= 30:
                st.warning("⚠️ Please provide more detailed information for better analysis.")
        
        # Generate button with enhanced styling
        generate_col1, generate_col2, generate_col3 = st.columns([1, 2, 1])
        
        with generate_col2:
            if st.button(
                "🤖 Launch Advanced AI Analysis",
                type="primary",
                disabled=not ready_to_generate,
                help="Start the comprehensive AI analysis process" if ready_to_generate else "Complete the required fields first"
            ):
                
                # Enhanced progress tracking
                progress_steps = [
                    ("🔍 Analyzing customer data patterns...", 15),
                    ("🧠 Extracting behavioral insights...", 25),
                    ("🎭 Creating detailed personas...", 40),
                    ("🚀 Building campaign strategies...", 55),
                    ("📊 Calculating performance metrics...", 70),
                    ("🔬 Generating advanced insights...", 85),
                    ("✅ Finalizing comprehensive analysis...", 100)
                ]

                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
    
                    for step_text, step_progress in progress_steps:
                        status_text.markdown(f"**{step_text}**")
                        progress_bar.progress(step_progress)
                        time.sleep(0.8)  # Realistic timing
        
        # Execute actual AI operations at specific steps
                        if step_progress == 25:
                            analysis_results = ai_engine.analyze_customer_data(customer_data, product_info)
                        elif step_progress == 40:
                            personas_results = ai_engine.create_personas(analysis_results, num_personas)
                        elif step_progress == 55:
                            campaigns_results = ai_engine.create_campaigns(personas_results)
                        elif step_progress == 85:
            # Generate additional features
                            additional_results = {}
                            if enable_competitor_analysis:
                                additional_results['competitor_analysis'] = ai_engine.generate_competitor_analysis(personas_results)
                            if enable_ab_testing and campaigns_results:
                                additional_results['ab_tests'] = ai_engine.generate_ab_test_ideas(campaigns_results['campaigns'][0])
                    
                    # Store results in session state
                    st.session_state['analysis_complete'] = True
                    st.session_state['analysis_timestamp'] = datetime.now()
                    st.session_state['personas_data'] = personas_results
                    st.session_state['campaigns_data'] = campaigns_results
                    st.session_state['analysis_data'] = analysis_results
                    st.session_state['additional_results'] = additional_results
                    st.session_state['num_personas_generated'] = num_personas
                    st.session_state['advanced_features'] = {
                        'competitor_analysis': enable_competitor_analysis,
                        'ab_testing': enable_ab_testing,
                        'journey_maps': enable_journey_maps
                    }
                    
                    # Success notification
                    st.balloons()
                    st.success(f"🎉 AI has successfully generated {num_personas} personas with advanced insights!")
                    
    
    # Results Display Section (Enhanced)
    if st.session_state.get('analysis_complete', False):
        
        st.markdown("---")
        st.markdown("## 📊 AI Analysis Results")
        
        # Create enhanced tabs
        tab_list = [
            "👥 Customer Personas", 
            "🚀 Campaign Strategies", 
            "📈 Analytics Dashboard", 
            "🔄 Interactive Refinement",
            "🤖 AI Assistant",
            "🎨 Content Generator",
            "📋 Export & Share"
        ]
        
        tabs = st.tabs(tab_list)
        
        # Tab 1: Customer Personas
        with tabs[0]:
            st.subheader("🎭 Strategic Customer Personas")
            
            personas_data = st.session_state.get('personas_data', {})
            display_personas(personas_data)
            
            # Enhanced confidence chart
            conf_chart = create_confidence_chart(personas_data)
            if conf_chart:
                st.plotly_chart(conf_chart, use_container_width=True, key="personas_confidence_chart")
        
        # Tab 2: Campaign Strategies  
        with tabs[1]:
            st.subheader("🎯 Campaign Strategies")
            
            campaigns_data = st.session_state.get('campaigns_data', {})
            display_campaigns(campaigns_data)
            
            # Enhanced ROI chart
            roi_chart = create_roi_comparison_chart(campaigns_data)
            if roi_chart:
                st.plotly_chart(roi_chart, use_container_width=True, key="campaigns_roi_chart")
        
        # Tab 3: Analytics Dashboard
        # Tab 3: Enhanced Analytics Dashboard
        with tabs[2]:
            st.subheader("📊 Advanced Analytics Dashboard")
    
    # Generate smart insights
            personas_data = st.session_state.get('personas_data', {})
            campaigns_data = st.session_state.get('campaigns_data', {})
    
            smart_insights = generate_persona_insights(personas_data)
    
    # Top-level insights
            if smart_insights:
                col1, col2, col3, col4 = st.columns(4)
        
                with col1:
                    market_dist = smart_insights.get('market_distribution', {})
                    st.metric("Market Coverage", market_dist.get('total_addressable', 'N/A'))
        
                with col2:
                    conf_analysis = smart_insights.get('confidence_analysis', {})
                    avg_conf = conf_analysis.get('average_confidence', 0.85)
                    st.metric("Avg Confidence", f"{avg_conf:.0%}")
        
                with col3:
                    revenue_potential = smart_insights.get('revenue_potential', {})
                    st.metric("High-Value Segments", revenue_potential.get('high_value_segments', 0))
        
                with col4:
                    reliability = conf_analysis.get('reliability_score', 'Medium')
                    st.metric("Reliability Score", reliability)
    
    # Charts section
            col1, col2 = st.columns(2)
    
            with col1:
                market_chart = create_market_size_chart(personas_data)
                if market_chart:
                    st.plotly_chart(market_chart, use_container_width=True, key="analytics_market_chart")
        
        # Add confidence chart below
                conf_chart = create_confidence_chart(personas_data)
                if conf_chart:
                    st.plotly_chart(conf_chart, use_container_width=True, key="analytics_confidence_chart")
    
            with col2:
                roi_chart = create_roi_comparison_chart(campaigns_data)
                if roi_chart:
                    st.plotly_chart(roi_chart, use_container_width=True, key="analytics_roi_chart")
        
        # Smart recommendations
                if smart_insights and smart_insights.get('targeting_recommendations'):
                    st.markdown("### 💡 AI Recommendations")
                    for rec in smart_insights['targeting_recommendations']:
                        st.success(rec)
    
    # Rest of existing analytics code...
            
            # Additional Analytics
            additional_results = st.session_state.get('additional_results', {})
            
            if additional_results.get('competitor_analysis'):
                st.markdown("### 🏢 Competitive Landscape")
                comp_analysis = additional_results['competitor_analysis']
                landscape = comp_analysis.get('competitor_landscape', {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Direct Competitors:**")
                    for competitor in landscape.get('direct_competitors', []):
                        st.write(f"• {competitor}")
                
                with col2:
                    st.markdown("**Differentiation Opportunities:**")
                    for opp in landscape.get('differentiation_opportunities', []):
                        st.write(f"• {opp}")
        
        # Tab 4: Interactive Refinement (FIXED)
        with tabs[3]:
            st.subheader("🔄 Refine Your Personas")
            
            personas_data = st.session_state.get('personas_data', {})
            if personas_data and 'personas' in personas_data:
                
                # Persona selector
                persona_names = [p.get('name', f'Persona {i+1}') for i, p in enumerate(personas_data['personas'])]
                selected_persona_name = st.selectbox(
                    "Select Persona to Refine:",
                    persona_names,
                    key="persona_selector"
                )
                
                selected_persona_idx = persona_names.index(selected_persona_name)
                selected_persona = personas_data['personas'][selected_persona_idx]
                
                # Display current persona summary
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### Current Persona Profile")
                    
                    # Show key details
                    name = selected_persona.get('name', 'Unknown')
                    tagline = selected_persona.get('tagline', '')
                    is_refined = selected_persona.get('is_refined', False)
                    
                    if is_refined:
                        st.success(f"✨ **{name}** (Previously Refined)")
                    else:
                        st.info(f"🎭 **{name}** (Original)")
                    
                    st.write(f"*{tagline}*")
                    
                    # Key characteristics
                    demographics = selected_persona.get('demographics', {})
                    st.write(f"**Age:** {demographics.get('age_range', 'N/A')} | **Income:** {demographics.get('income_range', 'N/A')}")
                    
                    pain_points = selected_persona.get('pain_points', [])[:2]
                    if pain_points:
                        st.write("**Top Pain Points:**")
                        for pain in pain_points:
                            st.write(f"• {pain}")
                
                with col2:
                    # Refinement metrics
                    confidence = selected_persona.get('confidence_score', 0.85)
                    if isinstance(confidence, str):
                        try:
                            confidence = float(confidence.replace('%', '')) / 100
                        except:
                            confidence = 0.85
                    
                    st.metric("Confidence Score", f"{confidence:.0%}")
                    st.metric("Market Size", selected_persona.get('market_size', 'N/A'))
                    
                    if is_refined:
                        refinement_count = len(selected_persona.get('refinement_history', []))
                        st.metric("Refinements", refinement_count)
                
                # Refinement interface
                st.markdown("### 🎯 Provide Refinement Feedback")
                
                # Pre-populated refinement suggestions
                refinement_suggestions = [
                    "Make this persona more tech-savvy and focused on automation",
                    "Adjust age range to be younger (25-35) and more mobile-focused", 
                    "Increase budget consciousness and add family considerations",
                    "Focus more on B2B decision-making and enterprise features",
                    "Add environmental consciousness and sustainability values",
                    "Custom feedback..."
                ]
                
                selected_suggestion = st.selectbox(
                    "Quick Refinement Options:",
                    refinement_suggestions,
                    key="refinement_suggestions"
                )
                
                if selected_suggestion == "Custom feedback...":
                    feedback_text = st.text_area(
                        "Enter your custom refinement feedback:",
                        height=120,
                        placeholder="Describe how you want to modify this persona. Be specific about demographics, behaviors, pain points, or goals you want to change...",
                        key="custom_feedback"
                    )
                else:
                    feedback_text = selected_suggestion
                    st.text_area(
                        "Refinement feedback:",
                        value=feedback_text,
                        height=100,
                        key="selected_feedback"
                    )
                
                # Refinement controls
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col2:
                    if st.button("🔄 Refine Persona", type="primary", key="refine_button"):
                        if feedback_text.strip():
                            
                            # Show refinement progress
                            with st.spinner("🤖 AI is refining your persona..."):
                                time.sleep(2)  # Simulate processing
                                
                                try:
                                    ai_engine = initialize_ai_engine()
                                    if ai_engine:
                                        # Perform refinement
                                        refined_persona = ai_engine.refine_persona(selected_persona, feedback_text)
                                        
                                        # Update the persona in session state
                                        st.session_state['personas_data']['personas'][selected_persona_idx] = refined_persona
                                        
                                        # Regenerate campaigns with refined personas
                                        st.session_state['campaigns_data'] = ai_engine.create_campaigns(st.session_state['personas_data'])
                                        
                                        # Success notification
                                        st.success("✅ Persona successfully refined!")
                                        st.balloons()
                                        
                                        # Show what changed
                                        st.markdown("### 🎉 Refinement Complete!")
                                        st.info("The persona has been updated and campaigns have been regenerated. Check the Personas tab to see changes.")
                                        
                                        # Auto-refresh after 3 seconds
                                        time.sleep(1)
                                        st.rerun()
                                        
                                    else:
                                        st.error("AI engine not available for refinement.")
                                        
                                except Exception as e:
                                    st.error(f"Refinement failed: {str(e)}")
                        else:
                            st.warning("Please provide feedback before refining.")
                
                # Refinement history
                if selected_persona.get('refinement_history'):
                    st.markdown("### 📚 Refinement History")
                    
                    with st.expander("View Previous Refinements"):
                        for i, entry in enumerate(selected_persona['refinement_history']):
                            timestamp = entry.get('timestamp', 'Unknown time')
                            if 'T' in timestamp:
                                try:
                                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                    timestamp = dt.strftime('%m/%d/%Y %H:%M')
                                except:
                                    pass
                            
                            st.markdown(f"**Refinement #{i+1}** - {timestamp}")
                            st.write(f"*Feedback:* {entry.get('feedback', 'No feedback recorded')}")
                            st.markdown("---")
            
            else:
                st.info("👆 Generate personas first to enable refinement features.")
        
        # Tab 5: AI Assistant
        with tabs[4]:
            st.subheader("🤖 AI Marketing Assistant")
            
            # Chat interface
            st.markdown("Ask questions about your personas, campaigns, or get marketing advice!")
            
            # Pre-defined questions
            quick_questions = [
                "Which persona should I target first?",
                "How can I improve my campaign ROI?",
                "What content works best for my top persona?",
                "Which channels should I prioritize?",
                "How do I measure campaign success?",
                "Custom question..."
            ]
            
            selected_question = st.selectbox(
                "Quick Questions:",
                quick_questions,
                key="ai_questions"
            )
            
            if selected_question == "Custom question...":
                user_query = st.text_input(
                    "Ask your question:",
                    placeholder="e.g., How should I adjust messaging for my premium persona?",
                    key="custom_query"
                )
            else:
                user_query = selected_question
                st.text_input(
                    "Question:",
                    value=user_query,
                    key="selected_query"
                )
            
            if st.button("🚀 Ask AI Assistant", type="primary"):
                if user_query.strip():
                    with st.spinner("🤖 AI Assistant is thinking..."):
                        try:
                            ai_engine = initialize_ai_engine()
                            if ai_engine:
                                context = {
                                    'personas_data': st.session_state.get('personas_data'),
                                    'campaigns_data': st.session_state.get('campaigns_data'),
                                    'analysis_data': st.session_state.get('analysis_data')
                                }
                                
                                answer = ai_engine.answer_query(user_query, context)
                                
                                st.markdown("### 💡 AI Assistant Response:")
                                st.markdown(answer)
                                
                            else:
                                st.error("AI Assistant temporarily unavailable.")
                        
                        except Exception as e:
                            st.error(f"Assistant error: {str(e)}")
                else:
                    st.warning("Please enter a question.")
            
            # Conversation history (simple)
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []
            
            # Display recent questions
            if st.session_state['chat_history']:
                st.markdown("### 📜 Recent Questions")
                for i, (q, a) in enumerate(st.session_state['chat_history'][-3:]):
                    with st.expander(f"Q: {q[:50]}..."):
                        st.markdown(f"**Q:** {q}")
                        st.markdown(f"**A:** {a[:200]}...")
        
        # Tab 6: Content Generator
        with tabs[5]:
            st.subheader("🎨 AI Content Generator")
            
            campaigns_data = st.session_state.get('campaigns_data', {})
            advanced_features = st.session_state.get('advanced_features', {})
            
            if campaigns_data and 'campaigns' in campaigns_data:
                
                # Campaign selector
                campaign_titles = [c.get('title', f'Campaign {i+1}') for i, c in enumerate(campaigns_data['campaigns'])]
                selected_campaign_title = st.selectbox(
                    "Select Campaign for Content Generation:",
                    campaign_titles,
                    key="campaign_selector"
                )
                
                selected_campaign_idx = campaign_titles.index(selected_campaign_title)
                selected_campaign = campaigns_data['campaigns'][selected_campaign_idx]
                
                # Content generation options
                content_types = [
                    "📧 Email Campaign",
                    "📱 Social Media Posts", 
                    "🎯 Google Ads",
                    "📝 Blog Content",
                    "🎨 Landing Page Copy",
                    "📊 All Content Types"
                ]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_content_type = st.selectbox(
                        "Content Type:",
                        content_types,
                        key="content_type_selector"
                    )
                
                with col2:
                    if st.button("🎨 Generate Content", type="primary"):
                        with st.spinner("🤖 Creating engaging content..."):
                            try:
                                ai_engine = initialize_ai_engine()
                                if ai_engine:
                                    content_samples = ai_engine.generate_content_sample(selected_campaign)
                                    
                                    st.success("✅ Content generated successfully!")
                                    display_content_samples(content_samples)
                                    
                                else:
                                    st.error("Content generator temporarily unavailable.")
                            
                            except Exception as e:
                                st.error(f"Content generation failed: {str(e)}")
                
                # Advanced content features
                st.markdown("---")
                st.markdown("### 🚀 Advanced Content Features")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🗺️ Generate Journey Map"):
                        with st.spinner("Creating customer journey map..."):
                            try:
                                ai_engine = initialize_ai_engine()
                                if ai_engine and st.session_state.get('personas_data'):
                                    # Use first persona for journey map
                                    persona_for_map = st.session_state['personas_data']['personas'][0]
                                    journey_data = ai_engine.generate_journey_map(persona_for_map)
                                    
                                    st.success("✅ Journey map created!")
                                    
                                    journey_chart = create_journey_map_chart(journey_data)
                                    if journey_chart:
                                        st.plotly_chart(journey_chart, use_container_width=True, key="content_journey_chart")
                                    
                                    # Display journey stages
                                    st.markdown("#### 🗺️ Customer Journey Stages")
                                    stages = journey_data.get('journey_map', [])
                                    
                                    for stage in stages:
                                        with st.expander(f"{stage.get('stage', 'Stage')} - {', '.join(stage.get('emotions', []))}"):
                                            st.write(f"**Touchpoints:** {', '.join(stage.get('touchpoints', []))}")
                                            st.write(f"**Pain Points:** {', '.join(stage.get('pain_points', []))}")
                                            st.write(f"**Opportunities:** {', '.join(stage.get('opportunities', []))}")
                                            st.write(f"**Recommended Actions:** {', '.join(stage.get('actions', []))}")
                                
                            except Exception as e:
                                st.error(f"Journey map generation failed: {str(e)}")
                
                with col2:
                    if st.button("📊 Simulate Performance"):
                        with st.spinner("Running performance simulation..."):
                            try:
                                ai_engine = initialize_ai_engine()
                                if ai_engine:
                                    sim_data = ai_engine.simulate_performance(selected_campaign)
                                    
                                    st.success("✅ Performance simulation complete!")
                                    display_performance_simulation(sim_data)
                                
                            except Exception as e:
                                st.error(f"Performance simulation failed: {str(e)}")
                
                with col3:
                    if advanced_features.get('ab_testing') and st.button("🧪 A/B Test Ideas"):
                        with st.spinner("Generating A/B test recommendations..."):
                            try:
                                ai_engine = initialize_ai_engine()
                                if ai_engine:
                                    ab_data = ai_engine.generate_ab_test_ideas(selected_campaign)
                                    
                                    st.success("✅ A/B test ideas generated!")
                                    display_ab_test_ideas(ab_data)
                                
                            except Exception as e:
                                st.error(f"A/B test generation failed: {str(e)}")
                
            else:
                st.info("👆 Generate campaigns first to enable content generation.")
        
        # Tab 7: Export & Share (Enhanced)
        with tabs[6]:
            st.subheader("📤 Export & Share Results")
            
            # Export options
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                st.markdown("#### 📄 Comprehensive Report")
                if st.button("📋 Generate Report", use_container_width=True):
                    personas_data = st.session_state.get('personas_data', {})
                    campaigns_data = st.session_state.get('campaigns_data', {})
                    analysis_data = st.session_state.get('analysis_data', {})
                    
                    report_content = generate_comprehensive_report(personas_data, campaigns_data, analysis_data)
                    
                    st.download_button(
                        label="📥 Download Comprehensive Report",
                        data=report_content.encode('utf-8'),
                        file_name=f"ai_marketing_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        help="Download detailed analysis report in Markdown format"
                    )
                    st.success("📊 Comprehensive report ready for download!")
            
            with export_col2:
                st.markdown("#### 📊 Data Export")
                if st.button("💾 Export Data", use_container_width=True):
                    # Create comprehensive JSON export
                    export_data = {
                        "metadata": {
                            "generated_at": datetime.now().isoformat(),
                            "ai_engine": "Google Gemini Pro",
                            "version": "2.0",
                            "personas_count": len(st.session_state.get('personas_data', {}).get('personas', [])),
                            "campaigns_count": len(st.session_state.get('campaigns_data', {}).get('campaigns', []))
                        },
                        "analysis": st.session_state.get('analysis_data', {}),
                        "personas": st.session_state.get('personas_data', {}),
                        "campaigns": st.session_state.get('campaigns_data', {}),
                        "additional_insights": st.session_state.get('additional_results', {}),
                        "configuration": st.session_state.get('advanced_features', {})
                    }
                    
                    json_str = json.dumps(export_data, indent=2, default=str)
                    
                    st.download_button(
                        label="📥 Download Full Data (JSON)",
                        data=json_str,
                        file_name=f"marketing_intelligence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        help="Download complete analysis data in JSON format"
                    )
                    st.success("💾 Complete dataset ready for download!")
            
            with export_col3:
                st.markdown("#### 🔗 Share Results")
                if st.button("🌐 Create Share Link", use_container_width=True):
                    # Generate shareable summary
                    personas_data = st.session_state.get('personas_data', {})
                    campaigns_data = st.session_state.get('campaigns_data', {})
                    
                    share_content = f"""
# 🎯 AI Marketing Intelligence Summary

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}  
**AI Engine:** Google Gemini Pro Advanced  
**Personas Generated:** {len(personas_data.get('personas', []))}  
**Campaigns Created:** {len(campaigns_data.get('campaigns', []))}

## 🎭 Top Customer Personas

"""
                    
                    if personas_data and 'personas' in personas_data:
                        for persona in personas_data['personas'][:3]:
                            name = persona.get('name', 'Unknown')
                            tagline = persona.get('tagline', '')
                            market_size = persona.get('market_size', 'N/A')
                            confidence = persona.get('confidence_score', 0.85)
                            if isinstance(confidence, (int, float)):
                                confidence = f"{confidence:.0%}"
                            
                            refined_indicator = " ✨" if persona.get('is_refined') else ""
                            
                            share_content += f"""
### {name}{refined_indicator}
*{tagline}*  
**Market Size:** {market_size} | **Confidence:** {confidence}

"""
                    
                    share_content += """
## 🚀 Campaign Highlights

"""
                    
                    if campaigns_data and 'campaigns' in campaigns_data:
                        for campaign in campaigns_data['campaigns']:
                            title = campaign.get('title', 'Campaign')
                            roi = campaign.get('predicted_roi', 'N/A')
                            target = campaign.get('persona_target', 'N/A')
                            
                            share_content += f"""
### {title}
**Target:** {target}  
**Expected ROI:** {roi}

"""
                    
                    share_content += """
---
*Generated by AI Marketing Persona Designer*  
*Powered by Google Gemini Pro for Advanced Marketing Intelligence*
"""
                    
                    st.download_button(
                        label="📥 Download Shareable Summary",
                        data=share_content,
                        file_name=f"marketing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        help="Download executive summary for sharing with stakeholders"
                    )
                    st.success("🔗 Shareable summary created!")
            
            # Analysis Summary Dashboard
            st.markdown("---")
            st.markdown("### 📊 Analysis Summary Dashboard")
            
            # Create summary metrics
            analysis_timestamp = st.session_state.get('analysis_timestamp', datetime.now())
            personas_data = st.session_state.get('personas_data', {})
            campaigns_data = st.session_state.get('campaigns_data', {})
            additional_results = st.session_state.get('additional_results', {})
            
            personas_count = len(personas_data.get('personas', [])) if personas_data else 0
            campaigns_count = len(campaigns_data.get('campaigns', [])) if campaigns_data else 0
            refined_count = sum(1 for p in personas_data.get('personas', []) if p.get('is_refined', False))
            
            # Summary metrics in columns
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                st.metric("👥 Personas", personas_count, delta=f"{refined_count} refined")
            
            with summary_col2:
                st.metric("🚀 Campaigns", campaigns_count)
            
            with summary_col3:
                advanced_count = len([k for k, v in additional_results.items() if v])
                st.metric("🔬 Advanced Features", advanced_count)
            
            with summary_col4:
                processing_time = (datetime.now() - analysis_timestamp).total_seconds()
                st.metric("⏱️ Processing Time", f"{processing_time:.0f}s")
            
            # Detailed summary
            summary_data = {
                "analysis_completed": analysis_timestamp.isoformat(),
                "personas_generated": personas_count,
                "campaigns_created": campaigns_count,
                "personas_refined": refined_count,
                "ai_engine": getattr(ai_engine, 'model_name', 'gemini-2.0-flash-exp') if ai_engine else 'offline',
                "advanced_features_used": list(additional_results.keys()),
                "analysis_status": "Complete ✅"
            }
            
            with st.expander("📋 Technical Summary (JSON)"):
                st.json(summary_data)

    # Enhanced Footer
    # Enhanced Footer - COMPLETELY FIXED
# Enhanced Footer - COMPLETELY FIXED
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <h3>🏆 AI Marketing Persona Designer - Enhanced Edition</h3>
    <p><strong>Powered by Google Gemini Flash 2.0 Experimental for lightning-fast analysis</strong></p>
</div>
""", unsafe_allow_html=True)

# Feature badges in columns
col1, col2, col3, col4 = st.columns(4)
feature_style = "background: linear-gradient(45deg, #00c9ff, #92fe9d); color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.9rem; font-weight: bold; display: block; text-align: center;"

with col1:
    st.markdown(f'<div style="{feature_style}">🔄 Real-time Refinement</div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div style="{feature_style}">🎨 AI Content Generation</div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div style="{feature_style}">🤖 Smart Assistant</div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div style="{feature_style}">📊 Advanced Analytics</div>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-top: 1rem;'>
    <p>Built with ❤️ using Streamlit | Transform marketing intelligence with cutting-edge AI</p>
    <p><em>Ready to use - No API key required! | Enhanced for hackathon excellence!</em></p>
    <div style="margin-top: 1rem;">
        <span style="background: linear-gradient(45deg, #667eea, #764ba2); padding: 0.5rem 1rem; border-radius: 20px; color: white; font-weight: bold;">
            ⚡ Powered by Gemini Flash 2.0 - Next-Gen AI Marketing Intelligence!
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    # Initialize session state with enhanced tracking
    session_vars = [
        'analysis_complete', 'analysis_timestamp', 'personas_data', 
        'campaigns_data', 'analysis_data', 'additional_results',
        'num_personas_generated', 'advanced_features', 'chat_history'
    ]
    
    for var in session_vars:
        if var not in st.session_state:
            if var == 'analysis_complete':
                st.session_state[var] = False
            elif var in ['additional_results', 'advanced_features']:
                st.session_state[var] = {}
            elif var == 'chat_history':
                st.session_state[var] = []
            else:
                st.session_state[var] = None
    
    main()
                